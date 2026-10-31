# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: one decode frame end to end, with the env it runs under (use the --interactive env).

--mode time: device time of the backbone decode trace, the fused tower trace and both back to back (replays with
    one sync at the end), then one real line (`close`) through FastSession.run with the card codec, timed per
    piece of the frame loop (host sampling + uploads, replays, readbacks, codec polls).
--mode prof (TT_METAL_DEVICE_PROFILER=1): every program of one backbone replay and one tower replay with its
    device kernel span, named by a graph capture of the same eager body; sums per op name and per layer kind.

    run_check.sh <log> tests.profile_frame7 --mode time --out /opt/ep0/port7/frame_time.json   (EXTRA_ENV=$INTER)
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def session(mesh, line, tower_mlp="bf16"):
    from models.demos.blackhole.drama3.host.generate_fast import FastSession

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ref = str(VOICE / "ref_v2.wav")
    refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
    s = FastSession(mesh, tower_mlp)
    rows = s.prompt(lines[line]["text"], refs)
    s._warm_T = rows.shape[0]
    s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
    s.warmup(rows)
    return s, rows


def dev_ms(mesh, fns, n=50):
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n):
        for f in fns:
            f()
    ttnn.synchronize_device(mesh)
    return (time.time() - t0) / n * 1e3


def mode_time(mesh, s, rows, out):
    from models.demos.blackhole.drama3.host.stream_codec import CodecProcess

    res = {}
    bb, tw = s.bb, s.tower
    rb = lambda: ttnn.execute_trace(mesh, bb.trace_id, cq_id=0, blocking=False)  # noqa: E731
    rt = lambda: ttnn.execute_trace(mesh, tw.fused_tid, cq_id=0, blocking=False)  # noqa: E731
    res["backbone_trace_ms"] = dev_ms(mesh, [rb])
    res["tower_trace_ms"] = dev_ms(mesh, [rt])
    res["bb_plus_tower_trace_ms"] = dev_ms(mesh, [rb, rt])
    # one real line, with the frame loop's pieces timed (wrappers around the calls FastSession.run makes)
    acc = defaultdict(float)
    cnt = defaultdict(int)

    def wrap(obj, name, key):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.time()
            r = f(*a, **k)
            acc[key] += time.time() - t0
            cnt[key] += 1
            return r

        setattr(obj, name, g)

    wrap(ttnn, "execute_trace", "execute_trace (enqueue)")
    wrap(ttnn, "to_torch", "to_torch (readback incl. wait)")
    wrap(ttnn, "copy_host_to_device_tensor", "host->device copies")
    wrap(ttnn, "from_torch", "from_torch")
    wrap(tw, "_step_fused_pre", "tower.step_fused total")
    wrap(tw, "prepare_frame", "tower.prepare_frame (noise)")
    wrap(bb, "decode_start", "bb.decode_start")
    wrap(bb, "decode_finish", "bb.decode_finish")
    cc = CodecProcess(threads=6, compile=True)
    try:
        t0 = time.time()
        frames, t, audio = s.run(rows, 0, 600, codec=cc)
        wall = time.time() - t0
    finally:
        cc.close()
    n = frames.shape[0]
    res["line"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
    res["line_wall_s"] = round(wall, 3)
    res["frames"] = n
    res["per_frame_ms"] = {k: round(v / max(n, 1) * 1e3, 3) for k, v in acc.items()}
    res["calls_per_frame"] = {k: round(c / max(n, 1), 2) for k, c in cnt.items()}
    gen_ms = t["gen_s"] / max(n, 1) * 1e3
    res["gen_ms_per_frame"] = round(gen_ms, 3)
    res["card_codec_ms_per_frame"] = round(t.get("card_codec_s", 0) / max(n, 1) * 1e3, 3)
    res = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def mode_prof(mesh, s, rows, out):
    from models.demos.blackhole.drama3.tests.profile_ops_dev import device_op_names, parse_log

    bb, tw = s.bb, s.tower
    T = rows.shape[0]
    bb._stage(s.embedder(rows[T - 1 : T])[0], T)
    names = {}

    def eager_bb():
        xn, lg, _ = bb._decode_body()
        ttnn.deallocate(xn)
        ttnn.deallocate(lg)

    def eager_tw():
        ttnn.deallocate(tw._fused_body())

    names["backbone"] = device_op_names(eager_bb)
    names["tower"] = device_op_names(eager_tw)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    order = []
    for name, tid in (("backbone", bb.trace_id), ("tower", tw.fused_tid)):
        for _ in range(2):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        time.sleep(0.01)
        ttnn.ReadDeviceProfiler(mesh)
        order.append(name)
    ttnn.close_device(mesh)
    spans = parse_log(len(order))
    res = {}
    for name, ops in zip(order, spans):
        nm = names.get(name) or []
        by = defaultdict(lambda: [0, 0.0])
        for i, (rid, st, d, rk) in enumerate(ops):
            k = nm[i] if len(nm) == len(ops) else "?"
            by[k][0] += 1
            by[k][1] += d
        span = (ops[-1][1] + ops[-1][2]) / 1e3 if ops else 0
        ksum = sum(o[2] for o in ops) / 1e3
        res[name] = dict(
            n_programs=len(ops), n_names=len(nm), span_ms=round(span, 3), sum_kernel_ms=round(ksum, 3),
            gaps_ms=round(span - ksum, 3),
            by_op={k: dict(n=v[0], ms=round(v[1] / 1e3, 3)) for k, v in sorted(by.items(), key=lambda x: -x[1][1])},
            ops=[dict(i=i, name=nm[i] if i < len(nm) else "?", start_us=round(st, 2), dur_us=round(d, 2)) for i, (rid, st, d, rk) in enumerate(ops)],
        )
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "ops"}, indent=1), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="time")
    ap.add_argument("--line", default="close")
    ap.add_argument("--tower-mlp", default="bf16", help="as generate_fast (--interactive: bfp8)")
    ap.add_argument("--out", default="/opt/ep0/port7/frame_time.json")
    a = ap.parse_args()
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=int(os.environ.get("DRAMA3_L1_SMALL", 65536)), trace_region_size=256 << 20)
    closed = False
    try:
        s, rows = session(mesh, a.line, a.tower_mlp)
        if a.mode == "time":
            mode_time(mesh, s, rows, a.out)
        else:
            mode_prof(mesh, s, rows, a.out)
            closed = True
    finally:
        if not closed:
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
