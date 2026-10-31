# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 TTS, port6: continuous batching on top of the Day 5b batch traces (opt-in, host/batch.py's BatchState).

B decode slots; a job (one line) is admitted into a free slot when it has arrived: its suffix is prefilled on the
one-row path in the slot's page-table row, its GDN state is copied on the card into the slot's row of the batched
state (kernels/row_copy.cpp: the other slots' rows stay as they are), and it decodes from the next step on, while
the other slots keep going. A slot freed by EOS takes the next job. Idle slots keep stepping at a frozen position
(their output is dropped). Admission policy (--policy): "warm-first" admits a new job only when every active
stream has its first audio window (FIRST_WINDOW frames), so a new caller's prefill never delays another caller's
first audio; "guard:<s>" also waits until every playing stream has s seconds of decoded audio ahead of its
playback (a prefill stalls every stream for ~0.4 s); "greedy" admits whenever a slot is free.

Needs DRAMA3_GDN_FUSED_BATCH=1 (the v4 GDN decode and its [1, 2B, D] partial-sum layout).

    run_gen.sh <log> --cbatch --batch-size 4 --lines all --arrivals 0 --out <dir>
    --arrivals: "0" (all at t = 0), "every:1.5" (one job every 1.5 s), "poisson:0.5:7" (rate per s, seed)
"""

import argparse
import json
import os
import queue
import time
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, log, reference_codes

KDIR = "models/demos/blackhole/drama3/kernels"


class RowCopy:
    """dst <- src on the card, data movement only (kernels/row_copy.cpp), spread over n_cores cores.
    mode 0: n whole tiles, src pages p0.. -> dst pages d0..; mode 1: the row-pair chunk of every src tile -> rows
    (dr, dr + 1) of the same dst tile."""

    def __init__(self, mesh, n_cores=16):
        g = mesh.compute_with_storage_grid_size()
        self.mesh = mesh
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, src, dst, mode, n, p0=0, d0=0, dr=0):
        ct = [mode] + list(ttnn.TensorAccessorArgs(src).get_compile_time_args()) + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        per = -(-n // len(self.cores))
        for i, c in enumerate(self.cores):
            i0 = min(i * per, n)
            rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), i0, min(per, n - i0), p0, d0, dr]
        kern = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/row_copy.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs, compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [ttnn.CBDescriptor(total_size=2048, core_ranges=self.crs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])]
        ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))


class SlotStreams:
    """The multi-stream codec process (CodecProcess(multi=True)) with job ids started on admission; each window
    latent is decoded on the card between steps. Keeps every chunk's arrival time (for the playback-underrun check)."""

    def __init__(self, cp, card, sr):
        self.cp, self.card, self.sr = cp, card, sr
        self.chunks, self.chunk_t, self.card_s, self.decode_s, self.done_t, self.samples = {}, {}, {}, {}, {}, {}

    def start(self, sid):
        self.chunks[sid], self.chunk_t[sid], self.card_s[sid], self.samples[sid] = [], [], 0.0, 0
        self.cp.inq.put(("start", sid))

    def push(self, sid, codes):
        self.cp.inq.put(("frame", sid, codes.numpy()))

    def lead(self, sid, now):
        """Decoded audio ahead of the playback position (playback from the first chunk); None before it."""
        ct = self.chunk_t[sid]
        return ct[0][0] + self.samples[sid] / self.sr - now if ct else None

    def end(self, sid):
        self.cp.inq.put(("end", sid))

    def _take(self, m):
        if m[0] == "z":
            _, sid, z, keep_from, n = m
            t0 = time.time()
            a = self.card.decode(z, keep_from, n)
            self.chunks[sid].append(a)
            self.card_s[sid] += time.time() - t0
            self.chunk_t[sid].append((time.time(), a.shape[0]))
            self.samples[sid] += a.shape[0]
        else:
            _, sid, self.decode_s[sid] = m
            self.done_t[sid] = time.time()

    def poll(self):
        while True:
            try:
                m = self.cp.outq.get_nowait()
            except queue.Empty:
                return
            self._take(m)

    def wait_done(self, sids):
        while not all(s in self.done_t for s in sids):
            self._take(self.cp.outq.get(timeout=600))


class DeviceSlotStreams:
    """SlotStreams with the quantizer on the card too (DRAMA3_QUANT_DEVICE=1): one stream_codec.DeviceQuantStream per
    job, each window decoded in the generation loop when its last frame is pushed (no codec process)."""

    def __init__(self, card, sr):
        self.card, self.sr = card, sr
        self.st, self.done_t = {}, {}

    def start(self, sid):
        from models.demos.blackhole.drama3.host.stream_codec import DeviceQuantStream

        self.st[sid] = DeviceQuantStream(self.card)

    def push(self, sid, codes):
        self.st[sid].push(codes)

    def end(self, sid):
        self.st[sid].finish()
        self.done_t[sid] = time.time()

    def poll(self):
        pass

    def wait_done(self, sids):
        pass

    def lead(self, sid, now):
        x = self.st[sid]
        return x.chunk_t[0][0] + sum(n for _, n in x.chunk_t) / self.sr - now if x.chunk_t else None

    @property
    def chunks(self):
        return {k: v.chunks for k, v in self.st.items()}

    @property
    def chunk_t(self):
        return {k: v.chunk_t for k, v in self.st.items()}

    @property
    def card_s(self):
        return {k: v.card_s for k, v in self.st.items()}


def arrivals(spec, n):
    if spec in ("0", "", None):
        return [0.0] * n
    kind, *p = spec.split(":")
    if kind == "every":
        return [i * float(p[0]) for i in range(n)]
    if kind == "poisson":
        g = torch.Generator().manual_seed(int(p[1]) if len(p) > 1 else 0)
        gaps = torch.empty(n).exponential_(float(p[0]), generator=g)
        gaps[0] = 0.0
        return torch.cumsum(gaps, 0).tolist()
    raise ValueError(spec)


def run_continuous(s, st, jobs, cc, max_frames, policy="warm-first", temperature=1.0, top_p=0.9, top_k=None):
    """jobs: [dict(id, rows, seed, arrival)] -> per job (frames, timing, audio)."""
    from models.demos.blackhole.drama3.host.batch import _draw, _draw_many, _pre
    from models.demos.blackhole.drama3.host.stream_codec import first_window, playback_check

    B, tw, bb, P = st.B, s.tower, s.bb, s.P
    rc = RowCopy(s.mesh)
    codec = DeviceSlotStreams(s.card_codec, cc.sr) if getattr(s.card_codec, "dq", None) is not None else SlotStreams(cc, s.card_codec, cc.sr)
    fw = first_window()
    pending = sorted(jobs, key=lambda j: j["arrival"])
    slot = [None] * B
    res = {}
    rec_pages = 32 * 16
    pos = torch.full((B,), s._warm_T - 1, dtype=torch.long)
    last_frame = [torch.zeros(11, dtype=torch.long)] * B
    noise = torch.ones(1, len(tw.residual), B, st.K)
    t_bb = t_tw = t_host = t_admit = 0.0
    t_draw = t_codec = 0.0  # port10: parts of t_host (sampling draws, codec windows on the card)
    vdraw = os.environ.get("DRAMA3_CB_VDRAW", "0") == "1"
    dmask = os.environ.get("DRAMA3_MOE_DEDUP_MASK", "0") == "1"
    last_act = None
    from models.demos.blackhole.drama3.tt import moe_fused_decode
    steps = 0
    active_steps = []  # active streams per step
    t_start = time.time()

    def admit(b, job):
        nonlocal pos
        rows = job["rows"]
        t0 = time.time()
        timing = dict(prompt_rows=rows.shape[0], prefix_rows=P, slot=b, batch=B, arrival_s=job["arrival"],
                      admit_s=t0 - t_start, queue_s=t0 - t_start - job["arrival"])
        st.stream_table(b)
        if P and (s.voice is None or not torch.equal(s.voice, rows[: P + 1])):
            st.stream_table(0)  # the prefix blocks are stream 0's (shared, read only for the others)
            t1 = time.time()
            s.set_voice(rows)
            timing["voice_prefill_s"] = time.time() - t1
            st.stream_table(b)
        normed, lg = s._prefill_line(rows)
        st.stream_table(None)
        t1 = time.time()
        for (dn, rb, cb, sp), (rec, conv, split) in zip(st.gdn, st.solo_gdn):
            rc(rec, rb, 0, rec_pages, d0=b * rec_pages)
            for q_solo, q_b in zip(split, sp):
                nq = q_solo.shape[-1] // 32  # port10: row 2b sits in tile row 2b // 32 (B > 16)
                rc(q_solo, q_b, 1, nq, d0=(2 * b // 32) * nq, dr=(2 * b) % 32)
        d = ttnn.to_torch(st.dec_out[0]).float().reshape(1, B, -1)
        d[0, b] = normed[-1]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(d.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), st.dec_out[0])
        pos[b] = rows.shape[0] - 1
        st.stage_pos(pos)  # the next tower step reads the positions the last backbone step staged: restage all slots
        ttnn.synchronize_device(s.mesh)
        timing["state_copy_s"] = time.time() - t1
        timing["prefill_s"] = time.time() - t0
        timing["prefill_backbone_s"], timing["prefill_tower_s"] = s.prefill_split
        gen = torch.Generator().manual_seed(job["seed"])
        codec.start(job["id"])
        slot[b] = dict(job=job, gen=gen, logits=lg, window=[], frames=[], done=False, ras=0, timing=timing,
                       pre=_pre(tw, rows[rows.shape[0] - 1], gen), first_frame=None, steps=0)

    while pending or any(x is not None for x in slot):
        now = time.time() - t_start
        free = [b for b in range(B) if slot[b] is None]
        warming = any(x is not None and len(x["frames"]) < fw for x in slot)
        hold = policy != "greedy" and warming
        if policy.startswith("guard:") and not hold:  # every playing stream needs this much decoded lead first
            g = float(policy.split(":")[1])
            leads = [codec.lead(x["job"]["id"], time.time()) for x in slot if x is not None]
            hold = any(ld is None or ld < g for ld in leads)
        if free and pending and pending[0]["arrival"] <= now and not hold:
            t0 = time.time()
            admit(free[0], pending.pop(0))
            t_admit += time.time() - t0
            continue  # admit the next one at once if the policy allows it
        act = [b for b in range(B) if slot[b] is not None]
        if not act:
            time.sleep(max(0.0, pending[0]["arrival"] - (time.time() - t_start)))
            continue
        # ---- tower step (all slots; idle ones repeat their last position, output dropped)
        t0 = time.time()
        host_rows = torch.zeros(st.n_host, B, tw.H)
        mains, code0s = [0] * B, [0] * B
        t1 = time.time()
        if vdraw:  # port10: every active stream's draw in one batched pass (same math and noise per row)
            xs = [slot[b] for b in act]
            ms_, cs_, rd_, rows_ = _draw_many(tw, [x["logits"] for x in xs], [x["window"] for x in xs], [x["pre"] for x in xs],
                                              temperature, top_p, top_k)
            for i, b in enumerate(act):
                mains[b], code0s[b] = ms_[i], cs_[i]
                xs[i]["ras"] += int(rd_[i])
                noise[0, :, b] = xs[i]["pre"]["heads"][:, : st.K]
            host_rows[:, act] = rows_
        for b in act if not vdraw else ():
            x = slot[b]
            main, code0, redrawn, rows5 = _draw(tw, x["logits"], x["window"], x["pre"], temperature, top_p, top_k)
            mains[b], code0s[b] = main, code0
            x["ras"] += int(redrawn)
            host_rows[:, b] = rows5
            noise[0, :, b] = x["pre"]["heads"][:, : st.K]
        t_draw += time.time() - t1
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), st.noise_buf)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host_rows.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), st.e_host_buf)
        t_host += time.time() - t0
        t0 = time.time()
        ttnn.execute_trace(st.mesh, st.tw_tid, cq_id=0, blocking=False)
        codes = ttnn.to_torch(st.codes_out).float().reshape(B, -1)[:, : len(tw.residual)].round().long()
        t_tw += time.time() - t0
        steps += 1
        active_steps.append(len(act))
        t0 = time.time()
        for b in act:
            x = slot[b]
            x["steps"] += 1
            if x["first_frame"] is None:
                x["first_frame"] = time.time() - t_start
            if mains[b] == tw.host.eos or len(x["frames"]) >= max_frames:
                x["done"] = True
                x["stopped_by_eos"] = mains[b] == tw.host.eos
                x["gen_end"] = time.time() - t_start
                codec.end(x["job"]["id"])
                res[x["job"]["id"]] = x
                slot[b] = None  # freed: the next job can take it at the next loop
                continue
            frame = torch.cat([torch.tensor([mains[b], code0s[b]]), codes[b]])
            x["frames"].append(frame)
            t1 = time.time()
            codec.push(x["job"]["id"], frame[1:])
            t_codec += time.time() - t1
            x["window"] = (x["window"] + [mains[b]])[-16:]
            last_frame[b] = frame
        codec.poll()
        act = [b for b in range(B) if slot[b] is not None]
        for b in act:
            pos[b] += 1
        emb = s.embedder(torch.stack(last_frame))
        t_host += time.time() - t0
        t0 = time.time()
        if dmask:  # port10 DRAMA3_MOE_DEDUP_MASK=1: idle slots add no experts to the dedup MoE
            now_act = tuple(slot[b] is not None for b in range(B))
            if now_act != last_act:
                moe_fused_decode.set_active_rows(s.mesh, B, now_act)
                last_act = now_act
        st.stage(emb, pos)
        ttnn.execute_trace(st.mesh, st.bb_tid, cq_id=0, blocking=False)
        for b in act:
            slot[b]["pre"] = _pre(tw, last_frame[b], slot[b]["gen"])
        lg_all = ttnn.to_torch(st.dec_out[1]).float().reshape(B, -1)[:, : bb.n_cand]
        for b in act:
            slot[b]["logits"] = lg_all[b]
        t_bb += time.time() - t0
        if steps % 100 == 0:
            log(f"  step {steps}: {len(act)} active, backbone {t_bb / steps * 1e3:.1f} ms, tower {t_tw / steps * 1e3:.1f} ms, "
                f"host {t_host / steps * 1e3:.1f} ms per step, {len(pending)} pending")
    t_end = time.time()
    codec.wait_done(list(res))
    out = {}
    for sid, x in res.items():
        f = torch.stack(x["frames"]) if x["frames"] else torch.zeros(0, 11, dtype=torch.long)
        t = x["timing"]
        audio = torch.cat(codec.chunks[sid]) if codec.chunks[sid] else torch.zeros(0)
        ct = codec.chunk_t[sid]
        t_first = ct[0][0] if ct else None
        under, worst, ahead = playback_check(ct, cc.sr)
        arr = t_start + x["job"]["arrival"]
        audio_s = f.shape[0] * cc.frame_length / cc.sr
        done = codec.done_t[sid]
        t.update(
            frames=f.shape[0], stream_steps=x["steps"], ras_redraws=x["ras"], stopped_by_eos=x.get("stopped_by_eos"),
            first_frame_s=x["first_frame"] - x["job"]["arrival"], ttfa_s=(t_first - arr) if t_first else None,
            ttfa_from_admit_s=(t_first - t_start - t["admit_s"]) if t_first else None,
            gen_end_s=x["gen_end"] - x["job"]["arrival"], audio_done_s=done - arr, audio_s=audio_s,
            rtf=(done - arr) / audio_s if audio_s else None,
            rtf_decode=((x["gen_end"] + t_start) - (t_start + t["admit_s"] + t["prefill_s"])) / audio_s if audio_s else None,
            card_codec_s=codec.card_s[sid], underruns=under, worst_late_s=worst, min_ahead_s=ahead,
            windows=len(ct),
        )
        out[sid] = (f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}, audio)
    summ = dict(B=B, steps=steps, wall_s=round(t_end - t_start, 3),
                backbone_ms_per_step=round(1e3 * t_bb / max(steps, 1), 2), tower_ms_per_step=round(1e3 * t_tw / max(steps, 1), 2),
                host_ms_per_step=round(1e3 * t_host / max(steps, 1), 2), draw_ms_per_step=round(1e3 * t_draw / max(steps, 1), 2),
                codec_ms_per_step=round(1e3 * t_codec / max(steps, 1), 2), admit_s_total=round(t_admit, 3),
                mean_active=round(sum(active_steps) / max(len(active_steps), 1), 2), policy=policy,
                start_epoch=t_start, end_epoch=t_end)
    return out, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="all")
    ap.add_argument("--lines-file", default="")
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="the line list this many times (more callers)")
    ap.add_argument("--arrivals", default="0")
    ap.add_argument("--policy", default="warm-first")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--codec-threads", type=int, default=6)
    a = ap.parse_args()
    if os.environ.get("DRAMA3_CB_THREADS"):  # port10: torch intra-op threads of the host loop (small tensors)
        torch.set_num_threads(int(os.environ["DRAMA3_CB_THREADS"]))
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["DRAMA3_KV_STREAMS"] = str(a.batch_size)
    os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"
    assert os.environ.get("DRAMA3_GDN_FUSED_BATCH") == "1" and a.batch_size > 1
    import yaml

    from models.demos.blackhole.drama3.host.batch import BatchState
    from models.demos.blackhole.drama3.host.codec import save_wav
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.drama3.host.stream_codec import CodecProcess
    from models.demos.blackhole.drama3.tt.device_check import dram_used_gb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.lines_file:
        lines = {l["id"]: dict(l, text=l["spoken"]) for l in json.load(open(a.lines_file))}
    else:
        lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ids = list(lines) if a.lines == "all" else a.lines.split(",")
    ref_text = Path(a.ref).with_suffix(".txt").read_text().strip()
    codes = reference_codes(a.ref, "/opt/ep0/day3")
    cc = CodecProcess(threads=a.codec_threads, compile=False, multi=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=int(os.environ.get("DRAMA3_L1_SMALL", 65536)), trace_region_size=256 << 20)
    try:
        s = FastSession(mesh)
        prompts = {lid: s.prompt(lines[lid]["text"], [(ref_text, codes)]) for lid in ids}
        s._warm_T = max(r.shape[0] for r in prompts.values())
        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + list(prompts.values()))
        s.warmup(prompts[ids[0]], capture=False)
        st = BatchState(s, a.batch_size)
        st.capture()
        if s.card_codec is not None:
            s.card_codec.capture()
        RowCopy(mesh)(st.solo_gdn[0][0], st.gdn[0][1], 0, 1)  # compile the copy programs before the timed run
        if s.P:  # the shared voice prefix, prefilled before the clock starts (as DRAMA3_PREWARM_VOICE in solo)
            st.stream_table(0)
            s.set_voice(prompts[ids[0]])
            st.stream_table(None)
            ttnn.synchronize_device(mesh)
        RowCopy(mesh)(st.solo_gdn[0][2][0], st.gdn[0][3][0], 1, 1)
        log(f"continuous batch {a.batch_size}: traces captured, card DRAM {dram_used_gb(mesh)[0]:.2f} GB")
        order = ids * a.repeat
        arr = arrivals(a.arrivals, len(order))
        jobs = [dict(id=f"{lid}" if a.repeat == 1 else f"{lid}_r{i // len(ids)}", rows=prompts[lid], seed=a.seed, arrival=arr[i])
                for i, lid in enumerate(order)]
        res, summ = run_continuous(s, st, jobs, cc, a.max_frames, policy=a.policy)
        timing = {"_summary": summ}
        audio_tot = 0.0
        for sid, (frames, t, audio) in res.items():
            save_wav(out / f"{sid}.wav", audio, cc.sr)
            torch.save(frames, out / f"{sid}.frames.pt")
            timing[sid] = t
            audio_tot += t["audio_s"]
            log(f"{sid}: {json.dumps(t)}")
        summ["audio_s"] = round(audio_tot, 3)
        summ["audio_per_wall"] = round(audio_tot / summ["wall_s"], 3)
        summ["dram_gb"], summ["dram_total_gb"] = (round(x, 2) for x in dram_used_gb(mesh))
        summ["arrivals"] = a.arrivals
        log(f"summary: {json.dumps(summ)}")
        json.dump(timing, open(out / "timing.json", "w"), indent=1)
    finally:
        ttnn.close_device(mesh)
        cc.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
