# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 bb12: DRAMA3_SAMPLER_FUSED draws (tt/device_sampler.FusedDraw) vs the lean sampler (DRAMA3_SAMPLER_LEAN) and
the host reference on the same logits + noise (random rows at several scales), the one-hot rows, and the device
time of 9 draws as one trace (lean vs fused).

    run_check.sh <log> tests.check_sampler_fused [--rows 2000] [--out /opt/ep0/port10/bb12/check_sampler_fused.json]
"""
import argparse
import json
import time

import torch

import ttnn
from models.demos.blackhole.drama3.tt import device_sampler as ds


def host_draw(logits, noise, top_k=30, top_p=0.9):
    srt, ids = torch.sort(logits.float(), descending=True)
    cum = torch.cumsum(torch.softmax(srt, -1), -1)
    drop = (cum > top_p) | (torch.arange(srt.numel()) >= top_k)
    drop[0] = False
    probs = torch.softmax(srt.masked_fill(drop, float("-inf")), -1)
    return int(ids[torch.argmax(probs / noise)])


def code_of(t):
    return int(round(float(ttnn.to_torch(t).float().reshape(-1)[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--out", default="/opt/ep0/port10/bb12/check_sampler_fused.json")
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    lean = ds.DeviceSampler(mesh)
    lean.lean = True
    fused = ds.FusedDraw(mesh, 1024, 30, 0.9)
    g = torch.Generator().manual_seed(0)
    res = dict(n=0, fused_vs_lean=0, fused_vs_host=0, lean_vs_host=0, onehot_bad=0, examples=[])
    t0 = time.time()
    for i in range(a.rows):
        x = torch.randn(1024, generator=g) * (0.5 + 6 * torch.rand(1, generator=g))
        noise = torch.empty(4096).exponential_(1, generator=g)
        xd = ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        nd = ttnn.from_torch(noise[: ds.K].reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        cl = code_of(lean(xd, nd)[0])
        cf_t, oh = fused(xd, nd)
        cf = code_of(cf_t)
        o = ttnn.to_torch(oh).float().reshape(-1)
        want = torch.zeros(1024)
        want[cf] = 1.0
        res["onehot_bad"] += int(not torch.equal(o, want))
        h = host_draw(torch.cat([x, torch.full((3072,), float("-inf"))]), noise)
        res["n"] += 1
        res["fused_vs_lean"] += int(cf != cl)
        res["fused_vs_host"] += int(cf != h)
        res["lean_vs_host"] += int(cl != h)
        if cf != cl and len(res["examples"]) < 8:
            res["examples"].append(dict(row=i, host=h, lean=cl, fused=cf))
        for t in (xd, nd, cf_t, oh):
            ttnn.deallocate(t)
    res["check_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)

    # 9 draws per trace (the tower's 9 heads), device-bound replays
    xs = [ttnn.from_torch(torch.randn(1, 1, 1, 1024) * 3, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh) for _ in range(9)]
    nb = ttnn.from_torch(torch.empty(1, 1, 9, ds.K).exponential_(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)

    def body(fn):
        out = []
        for r in range(9):
            nz = ttnn.slice(nb, [0, 0, r, 0], [1, 1, r + 1, ds.K])
            out.append(fn(xs[r], nz)[0])
        return ttnn.concat(out, dim=-1)

    times = {}
    for name, fn in (("lean", lean), ("fused", fused)):
        body(fn)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        codes = body(fn)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        n = 50
        t0 = time.time()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times[name] = round((time.time() - t0) / n * 1e3, 3)
        times[name + "_codes"] = ttnn.to_torch(codes).float().reshape(-1)[:9].tolist()
    res["trace_ms_9_draws"] = times
    res["traced_codes_equal"] = times["lean_codes"] == times["fused_codes"]
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    ttnn.close_mesh_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
