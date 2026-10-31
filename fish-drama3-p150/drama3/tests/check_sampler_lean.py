# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: DRAMA3_SAMPLER_LEAN draws vs the Day 4b device sampler on the same logits + noise (random rows and
scaled rows), plus the host reference (tower_device.sample_with_noise semantics for the heads)."""
import json

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


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    a = ds.DeviceSampler(mesh)
    b = ds.DeviceSampler(mesh)
    b.lean = True
    g = torch.Generator().manual_seed(0)
    res = dict(n=0, base_vs_host=0, lean_vs_host=0, lean_vs_base=0, examples=[])
    for i in range(200):
        x = torch.randn(1024, generator=g) * (0.5 + 4 * torch.rand(1, generator=g))
        noise = torch.empty(4096).exponential_(1, generator=g)
        xd = ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        nd = ttnn.from_torch(noise[: ds.K].reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        ca = int(round(float(ttnn.to_torch(a(xd, nd)[0]).float().reshape(-1)[0])))
        cb = int(round(float(ttnn.to_torch(b(xd, nd)[0]).float().reshape(-1)[0])))
        h = host_draw(torch.cat([x, torch.full((3072,), float("-inf"))]), noise)
        res["n"] += 1
        res["base_vs_host"] += int(ca != h)
        res["lean_vs_host"] += int(cb != h)
        res["lean_vs_base"] += int(cb != ca)
        if cb != ca and len(res["examples"]) < 5:
            res["examples"].append(dict(host=h, base=ca, lean=cb))
    # traced: persistent inputs, captured once per sampler, replayed per row
    xd = ttnn.from_torch(torch.zeros(1, 1, 1, 1024), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    nd = ttnn.from_torch(torch.ones(1, 1, 1, ds.K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    outs = {}
    for name, smp in (("base", a), ("lean", b)):
        smp(xd, nd)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        code, _ = smp(xd, nd)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        outs[name] = (tid, code)
    res["traced_mismatch"] = dict(base=0, lean=0)
    for i in range(50):
        x = torch.randn(1024, generator=g) * 2
        noise = torch.empty(4096).exponential_(1, generator=g)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), xd)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(noise[: ds.K].reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), nd)
        h = host_draw(torch.cat([x, torch.full((3072,), float("-inf"))]), noise)
        for name, (tid, code) in outs.items():
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            c = int(round(float(ttnn.to_torch(code).float().reshape(-1)[0])))
            res["traced_mismatch"][name] += int(c != h)
            if c != h and i < 3:
                print(name, "traced", c, "host", h, flush=True)
    print(json.dumps(res), flush=True)
    json.dump(res, open("/opt/ep0/port7/check_sampler_lean.json", "w"), indent=1)
    ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
