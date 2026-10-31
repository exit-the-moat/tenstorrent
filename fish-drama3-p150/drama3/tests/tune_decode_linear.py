# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: inventory + program-config sweep of every one-token ttnn.linear / ttnn.matmul in a backbone decode step
(one GDN layer, one attention layer, the final norm/head) and one tower step (all 13 layers + heads).

For each distinct (x shape, weight shape, weight dtype, out dtype, memory config, compute config) it times the
call as the model makes it (default program) and with MatmulMultiCoreReuseMultiCast1DProgramConfig candidates
(mcast in0, per_core_N / in0_block_w sweep), each as a one-op trace replayed N times, and reports max |diff| vs
the default output. Writes the best per signature to --out (json), which tt/decode_linear.py can load.

    (fast env) python -m models.demos.blackhole.drama3.tests.tune_decode_linear --out /opt/ep0/day4b/linear_tune.json
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes

REC = dict(on=False, calls=[])


def sig(x, w, kw):
    mc = kw.get("memory_config")
    ckc = kw.get("compute_kernel_config")
    return json.dumps(
        dict(
            x=list(x.shape), w=list(w.shape), wdt=str(w.dtype), xdt=str(x.dtype), out=str(kw.get("dtype")),
            mc=str(mc.buffer_type) if mc is not None else None, xbuf=str(x.memory_config().buffer_type),
            act=str(kw.get("activation")),
            ckc=(str(getattr(ckc, "math_fidelity", "?")), str(getattr(ckc, "fp32_dest_acc_en", "?"))) if ckc is not None else None,
        ),
        sort_keys=True,
    )


def install():
    for name in ("linear", "matmul"):
        fn = getattr(ttnn, name)

        def w(*args, _fn=fn, _name=name, **kw):
            if REC["on"] and len(args) >= 2 and isinstance(args[1], ttnn.Tensor) and args[0].shape[-2] == 1 and "program_config" not in kw:
                if len(args[1].shape) <= 2 or all(d == 1 for d in list(args[1].shape)[:-2]):
                    key = sig(args[0], args[1], kw)
                    seen = REC.setdefault("seen", {})
                    if key not in seen:  # keep a copy of the activation: the model frees it right after
                        seen[key] = ttnn.clone(args[0])
                    REC["calls"].append((_name, seen[key], args[1], dict(kw)))
            return _fn(*args, **kw)

        setattr(ttnn, name, w)
    REC["orig"] = {n: getattr(ttnn, n).__defaults__ for n in ("linear",)}


def candidates(Kt, Nt, grid):
    max_cores = grid.x * grid.y
    pns = sorted({math.ceil(Nt / c) for c in range(1, max_cores + 1)})
    out = []
    for pn in pns:
        if pn > 16:
            continue
        cores = math.ceil(Nt / pn)
        for ibw in (1, 2, 4, 8, 16):
            if Kt % ibw:
                continue
            sw = max(d for d in (1, 2, 4) if pn % d == 0)
            out.append(dict(pn=pn, ibw=ibw, sw=sw, cores=cores, gx=min(grid.x, cores), gy=math.ceil(cores / grid.x)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/day4b/linear_tune.json")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    os.environ["DRAMA3_TOWER_DEVICE_SAMPLING"] = "0"
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    grid = mesh.compute_with_storage_grid_size()
    try:
        s = FastSession(mesh, tower_mlp=os.environ.get("TUNE_TOWER_MLP", "bf16"))
        lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
        ref = str(VOICE / "ref_v2.wav")
        refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
        rows = s.prompt(lines["fix"]["text"], refs)
        s.P = 0
        normed, logits = s._prefill_line(rows)
        T = rows.shape[0]
        install()
        REC["on"] = True
        bb = s.bb
        bb._stage(s.embedder(rows[T - 1 : T])[0], T)
        xn, lg, _ = bb._decode_body()
        hd = s._row_dev(normed[-1])
        s.tower.step(logits, hd, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0))
        REC["on"] = False
        groups = {}
        for name, x, w, kw in REC["calls"]:
            groups.setdefault(sig(x, w, kw), []).append((name, x, w, kw))
        print(f"{len(REC['calls'])} one-token matmul calls, {len(groups)} signatures", flush=True)

        def timed(fn):
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(a.n):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ms = (time.time() - t0) / a.n * 1e3
            r = ttnn.to_torch(out).float()
            ttnn.release_trace(mesh, tid)
            return ms, r

        results = []
        for key, calls in groups.items():
            name, x, w, kw = calls[0]
            f = getattr(ttnn, name)
            f(x, w, **kw)  # compile default
            d_ms, ref = timed(lambda: f(x, w, **kw))
            Kt, Nt = math.ceil(w.shape[-2] / 32), math.ceil(w.shape[-1] / 32)
            best = None
            for c in candidates(Kt, Nt, grid):
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(c["gx"], c["gy"]), in0_block_w=c["ibw"], out_subblock_h=1, out_subblock_w=c["sw"],
                    per_core_M=1, per_core_N=c["pn"], fuse_batch=True, fused_activation=None, mcast_in0=True,
                )
                kw2 = dict(kw, program_config=pc)
                if "activation" in kw2 and kw2["activation"] is not None:
                    continue  # fused activation goes in the program config; keep those on the default
                try:
                    f(x, w, **kw2)
                    ms, r = timed(lambda: f(x, w, **kw2))
                except Exception as e:  # noqa: BLE001
                    continue
                diff = float((r - ref).abs().max())
                scale = float(ref.abs().max()) + 1e-6
                if diff / scale > 2e-2:
                    continue
                if best is None or ms < best["ms"]:
                    best = dict(c, ms=round(ms, 4), rel_diff=round(diff / scale, 5))
            row = dict(sig=json.loads(key), calls=len(calls), default_ms=round(d_ms, 4), best=best,
                       saved_ms_per_frame_est=round((d_ms - best["ms"]) * len(calls), 4) if best else 0)
            results.append(row)
            print(json.dumps(row), flush=True)
        results.sort(key=lambda r: -r["saved_ms_per_frame_est"])
        json.dump(results, open(a.out, "w"), indent=1)
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
