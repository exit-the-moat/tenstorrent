# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4 check of the paged + traced backbone against the Day 3 voice golden (close prompt + 24 teacher frames).

    python -m models.demos.blackhole.drama3.tests.check_paged_backbone [--golden /opt/ep0/day3/golden_voice]
"""
import argparse
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)

MODEL = "/models/drama-3"


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="/opt/ep0/day3/golden_voice")
    ap.add_argument("--rows", default="/opt/ep0/day3/close.rows_teacher24.pt")
    ap.add_argument("--prompt-rows", type=int, default=718)
    ap.add_argument("--out", default="/opt/ep0/day4/check_paged_backbone.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.generate import _Golden as Golden_load
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.device_check import dram_used_gb
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    emb = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])
    rows = torch.load(a.rows)
    E = emb(rows)
    T = a.prompt_rows
    gold = Golden_load(a.golden)
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    res = {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand, log=log)

        # 1) whole-prompt masked prefill, per layer vs golden, all rows incl. the tail
        bb.reset()
        t0 = time.time()
        normed, last, per_layer = bb.prefill(E[:T], collect=True)
        res["prefill_s_first"] = round(time.time() - t0, 2)
        lp = [pcc(x, gold.layers[i][:T]) for i, x in enumerate(per_layer)]
        rowmin = [min(pcc(x[r], gold.layers[i][r]) for i, x in enumerate(per_layer)) for r in range(T - 16, T)]
        ref_last = gold.cand_logits[T - 1]
        res.update(
            prefill_layer_pcc_min=round(min(lp), 6),
            prefill_layer_pcc_mean=round(sum(lp) / len(lp), 6),
            prefill_layer_pcc=[round(x, 6) for x in lp],
            prefill_final_norm_pcc=round(pcc(normed, gold.final_norm[:T]), 6),
            tail16_row_layer_pcc_min=[round(x, 5) for x in rowmin],
            last_logits_pcc=round(pcc(last, ref_last), 6),
            last_top1_match=int(last.argmax()) == int(ref_last.argmax()),
            last_top5_overlap=len(set(torch.topk(last, 5).indices.tolist()) & set(torch.topk(ref_last, 5).indices.tolist())),
        )
        log(json.dumps(res))
        del per_layer

        # 2) capture the decode trace (prefill programs are compiled already)
        t0 = time.time()
        bb.capture(pos=T)
        res["capture_s"] = round(time.time() - t0, 1)

        def run(mode):
            bb.reset()
            t0 = time.time()
            n, l, _ = bb.prefill(E[:T])
            t_pre = time.time() - t0
            outs, times = [], []
            for i in range(T, rows.shape[0]):
                t0 = time.time()
                if mode == "trace":
                    h, lg = bb.decode(E[i], i)
                else:
                    h, lg, _ = bb.decode_eager(E[i], i)
                times.append(time.time() - t0)
                outs.append((h, lg))
            return t_pre, outs, times

        for mode in ("trace", "eager"):
            t_pre, outs, times = run(mode)
            fn = [pcc(h, gold.final_norm[T + j]) for j, (h, _) in enumerate(outs)]
            lg = [pcc(l, gold.cand_logits[T + j]) for j, (_, l) in enumerate(outs)]
            top1 = sum(int(l.argmax()) == int(gold.cand_logits[T + j].argmax()) for j, (_, l) in enumerate(outs))
            res[mode] = dict(
                prefill_s=round(t_pre, 2),
                ms_per_step_median=round(1000 * sorted(times)[len(times) // 2], 1),
                ms_per_step_min=round(1000 * min(times), 1),
                final_norm_pcc_min=round(min(fn), 6),
                final_norm_pcc_mean=round(sum(fn) / len(fn), 6),
                logits_pcc_mean=round(sum(lg) / len(lg), 6),
                top1=f"{top1}/{len(outs)}",
            )
            res[mode + "_outs"] = outs
            log(f"{mode}: {json.dumps(res[mode])}")
        tr, ea = res.pop("trace_outs"), res.pop("eager_outs")
        res["trace_vs_eager_logits_maxdiff"] = round(max(float((a[1] - b[1]).abs().max()) for a, b in zip(tr, ea)), 6)
        log(f"trace vs eager max |logit diff| {res['trace_vs_eager_logits_maxdiff']}; DRAM {dram_used_gb(mesh)[0]:.2f} GB")
        json.dump(res, open(a.out, "w"), indent=1)
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
