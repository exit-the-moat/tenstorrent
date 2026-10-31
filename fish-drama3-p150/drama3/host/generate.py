# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 text-to-speech, end to end: backbone on one P150, codebook tower + sampling + codec on the host CPU.

Per line: reference-bracket-v1 prompt with the voice reference as semantic rows -> host-built mixed embeddings
-> card prefill (largest multiple of 32) + single-row decode of the prompt tail -> tower prefill over the
prompt rows -> per frame: tower step (sample main token with RAS, then codes 0..9) -> mixed embedding of the
frame -> card decode -> ... until end of speech -> codec decode -> 44.1 kHz WAV.

    /opt/ep0/run.sh "PYTHONPATH=\\$PYTHONPATH:/opt/ep0/pyext-codec <env> python -m models.demos.blackhole.drama3.host.generate \\
        --lines close --out /opt/ep0/day3/out"

--teacher-frames F: feed the frames in F (a .pt of [N, 11]) instead of sampling, and stop after them; with
--golden D the card's per-layer outputs are compared in-process with the CPU reference in D (cpu_layer_ref
--rows) for every prompt row and teacher-forced frame.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn

MODEL = "/models/drama-3"
VOICE = Path("/opt/ep0/voice")
TT_ENV = dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def reference_codes(wav, cache_dir):
    path = Path(cache_dir) / (Path(wav).stem + ".codes.pt")
    if path.exists():
        return torch.load(path)
    from models.demos.blackhole.drama3.host.codec import encode_reference

    t0 = time.time()
    codes = encode_reference(str(wav))
    torch.save(codes, path)
    log(f"encoded {wav}: codes {tuple(codes.shape)} in {time.time() - t0:.1f}s -> {path}")
    return codes


class Session:
    def __init__(self, mesh, max_seq_len):
        from tokenizers import Tokenizer

        from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
        from models.demos.blackhole.drama3.host.tower import CodebookTower
        from models.demos.blackhole.drama3.tt.backbone import Drama3Backbone
        from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
        from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

        self.config = json.load(open(f"{MODEL}/config.json"))
        self.tok = Tokenizer.from_file(f"{MODEL}/tokenizer.json")
        wmap = weight_map(MODEL)
        t0 = time.time()
        self.tower = CodebookTower(MODEL, wmap, self.config)
        emb = read_tensors(MODEL, wmap, ["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
        self.embedder = FrameEmbedder(
            emb, self.tower.codebook_embeddings, self.config["semantic_start_token_id"], self.config["semantic_end_token_id"]
        )
        log(f"host tower + embeddings loaded in {time.time() - t0:.0f}s")
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=max_seq_len, enable_mtp=False)
        self.backbone = Drama3Backbone(mesh, args, args.weight_cache_path(), wmap, self.tower.candidates, log=log)
        self.eos = self.config["eos_token_id"]

    def prompt(self, text, references):
        from models.demos.blackhole.drama3.host.prompt import build_prompt_rows

        return build_prompt_rows(text, references, self.tok)

    def run(self, rows, seed, max_frames, teacher=None, golden=None, temperature=1.0, top_p=0.9, top_k=None):
        """Generate one line. Returns (frames [N, 11], timing dict, check dict)."""
        T = rows.shape[0]
        P = (T // 32) * 32
        bb, tower = self.backbone, self.tower
        t = dict(prompt_rows=T, prefill_rows=P)
        check = {}
        t_start = time.time()
        emb = self.embedder(rows)
        bb.reset()
        tower.reset(T + max_frames + 8)
        gold = _Golden(golden) if golden else None

        t0 = time.time()
        normed_p, logits_p, per_layer = bb.prefill(emb[:P], collect=gold is not None)
        t["prefill_s"] = time.time() - t0
        if gold:
            check["prefill_layers"] = gold.compare_rows(per_layer, 0, P)
            check["prefill_final_norm"] = round(pcc(normed_p, gold.final_norm[:P]), 6)
        normed = [normed_p]
        last_logits = logits_p[-1]
        t0 = time.time()
        tail = []
        for p in range(P, T):  # the prompt tail, one row at a time
            n_row, last_logits, pl = bb.decode(emb[p], p, collect=gold is not None)
            normed.append(n_row[None])
            if gold:
                tail.append(gold.compare_decode(pl, n_row, last_logits, p))
        t["tail_s"] = time.time() - t0
        if gold:
            check["tail_rows"] = tail
        normed = torch.cat(normed)

        log(f"  card prefill {P} rows {t['prefill_s']:.1f}s, tail {T - P} rows {t['tail_s']:.1f}s")
        t0 = time.time()
        tower.prefill(normed[: T - 1], torch.arange(T - 1), rows[: T - 1], rows[1:T])
        t["tower_prefill_s"] = time.time() - t0
        log(f"  tower prefill {T - 1} rows {t['tower_prefill_s']:.1f}s")

        gen = torch.Generator().manual_seed(seed)
        frames, window, ras_hits = [], [], 0
        h, pos, inp = normed[T - 1], T - 1, rows[T - 1]
        t_dec = t_tow = 0.0
        t_first = None
        steps = []
        for i in range(max_frames if teacher is None else len(teacher)):
            t0 = time.time()
            frame, redrawn = tower.step(last_logits, h, pos, inp, window, gen, temperature, top_p, top_k)
            t_tow += time.time() - t0
            if teacher is not None:
                steps.append(dict(step=i, sampled_main=int(frame[0])))
                frame = teacher[i]
            ras_hits += int(redrawn)
            if t_first is None:
                t_first = time.time() - t_start
            if int(frame[0]) == self.eos:
                break
            frames.append(frame)
            window = (window + [int(frame[0])])[-16:]
            pos += 1
            t0 = time.time()
            h, last_logits, pl = bb.decode(self.embedder(frame[None])[0], pos, collect=gold is not None)
            t_dec += time.time() - t0
            if gold and steps and pos < gold.rows:
                steps[-1].update(gold.compare_decode(pl, h, last_logits, pos))
            inp = frame
            if (i + 1) % 25 == 0:
                log(f"  frame {i + 1}: backbone {t_dec / (i + 1) * 1e3:.0f} ms/frame, tower {t_tow / (i + 1) * 1e3:.0f} ms/frame")
        t.update(
            decode_backbone_s=t_dec,
            tower_steps_s=t_tow,
            frames=len(frames),
            ras_redraws=ras_hits,
            first_frame_s=t_first,
            stopped_by_eos=len(frames) < (max_frames if teacher is None else len(teacher)),
            gen_s=time.time() - t_start,
        )
        if steps:
            check["decode_steps"] = steps
        return (torch.stack(frames) if frames else torch.zeros(0, 11, dtype=torch.long)), t, check


class _Golden:
    """CPU reference (cpu_layer_ref --rows) outputs: per-layer residual (fp16), final norm, candidate logits."""

    def __init__(self, d):
        self.d = Path(d)
        self.layers = [torch.load(self.d / f"layer_{i:02d}.pt")["hidden"].float() for i in range(40)]
        self.final_norm = torch.load(self.d / "final_norm.pt")["hidden"].float()
        self.cand_logits = torch.load(self.d / "cand_logits.pt")["logits"].float()
        self.rows = self.final_norm.shape[0]

    def compare_rows(self, per_layer, r0, r1):
        return [round(pcc(x, self.layers[i][r0:r1]), 6) for i, x in enumerate(per_layer)]

    def compare_decode(self, per_layer, normed, logits, p):
        ref_l = self.cand_logits[p]
        return dict(
            row=p,
            layer_pcc_min=round(min(pcc(x, self.layers[i][p]) for i, x in enumerate(per_layer)), 6) if per_layer else None,
            final_norm_pcc=round(pcc(normed, self.final_norm[p]), 6),
            logits_pcc=round(pcc(logits, ref_l), 6),
            top1_match=int(logits.argmax()) == int(ref_l.argmax()),
            top5_overlap=len(set(torch.topk(logits, 5).indices.tolist()) & set(torch.topk(ref_l, 5).indices.tolist())),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="close", help="comma list of line ids from lines.yaml, or 'all'")
    ap.add_argument("--ref", default=str(VOICE / "ref_long.wav"))
    ap.add_argument("--out", default="/opt/ep0/day3/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--teacher-frames", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--save-rows", default=None, help="write the prompt rows (+ teacher frames) of the first line here")
    ap.add_argument("--no-codec", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ids = list(lines) if a.lines == "all" else a.lines.split(",")
    ref_text = Path(a.ref).with_suffix(".txt").read_text().strip()
    codes = reference_codes(a.ref, out.parent)
    teacher = torch.load(a.teacher_frames) if a.teacher_frames else None

    from models.demos.blackhole.drama3.host.codec import decode_codes, load_codec, save_wav
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    codec = load_codec()
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        s = Session(mesh, max_seq_len=4096)
        env = {k: os.environ.get(k) for k in os.environ if k.startswith("QWEN36_")}
        log(f"fidelity env {env}")
        for lid in ids:
            rows = s.prompt(lines[lid]["text"], [(ref_text, codes)])
            if a.save_rows:
                torch.save(rows if teacher is None else torch.cat([rows, teacher]), a.save_rows)
                log(f"saved {rows.shape[0]} prompt rows{'' if teacher is None else f' + {len(teacher)} teacher frames'} -> {a.save_rows}")
            t_line = time.time()
            frames, t, check = s.run(rows, a.seed, a.max_frames, teacher=teacher, golden=a.golden)
            t["codec_s"] = 0.0
            if frames.shape[0] and not a.no_codec:
                t0 = time.time()
                audio = decode_codes(frames[:, 1:].T)
                t["codec_s"] = time.time() - t0
                save_wav(out / f"{lid}{a.tag}.wav", audio, codec.sample_rate)
                t0 = time.time()
                decode_codes(frames[: min(8, len(frames)), 1:].T)  # first chunk of audio, for the latency
                t["ttfa_s"] = round(t["first_frame_s"] + (time.time() - t0), 3)
            t["compute_s"] = round(time.time() - t_line, 3)
            t["audio_s"] = round(frames.shape[0] * codec.frame_length / codec.sample_rate, 3)
            t["rtf"] = round(t["compute_s"] / t["audio_s"], 3) if t["audio_s"] else None
            t["start_epoch"], t["end_epoch"] = t_line, time.time()
            t = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}
            torch.save(frames, out / f"{lid}{a.tag}.frames.pt")
            timing[lid + a.tag] = t
            json.dump(timing, open(timing_path, "w"), indent=1)
            log(f"{lid}: {json.dumps(t)}")
            if check:
                json.dump(check, open(out / f"{lid}{a.tag}.check.json", "w"), indent=1)
                log(f"{lid} check: prefill layers min {min(check['prefill_layers']):.5f}, final norm {check['prefill_final_norm']}")
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
