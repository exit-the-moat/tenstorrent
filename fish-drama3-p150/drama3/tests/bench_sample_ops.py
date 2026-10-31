# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: which ttnn ops can the main-token sampler use on the card, and what do they cost (one-op traces)?
topk widths on a [1,1,1,4128] row, full sort, embedding gathers from a device index, non-blocking readback."""
import json
import time

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {}

    def timed(name, fn, n=200):
        try:
            fn()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(n):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.time() - t0) / n * 1e6
            ttnn.release_trace(mesh, tid)
            res[name] = round(us, 2)
            print(name, round(us, 2), flush=True)
            return out
        except Exception as e:  # noqa: BLE001
            res[name] = "ERR " + str(e).splitlines()[0][:200]
            print(name, res[name], flush=True)
            return None

    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, 4128) * 3
    for dt_name, dt in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        xd = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
        for k in (32, 64, 128, 256, 512):
            o = timed(f"topk{k}_{dt_name}", lambda: ttnn.topk(xd, k, dim=-1, largest=True, sorted=True))
            if o is not None:
                v = ttnn.to_torch(o[0]).float().reshape(-1)[:k]
                ref = torch.topk(x.reshape(-1).to(torch.bfloat16 if dt_name == "bf16" else torch.float32).float(), k).values
                res[f"topk{k}_{dt_name}_maxerr"] = float((v - ref).abs().max())
        timed(f"sort_{dt_name}", lambda: ttnn.sort(xd, dim=-1, descending=True))
        timed(f"exp_{dt_name}", lambda: ttnn.exp(xd))
        timed(f"sum_{dt_name}", lambda: ttnn.sum(xd, dim=-1, keepdim=True))
        timed(f"cumsum64_{dt_name}", lambda: ttnn.cumsum(ttnn.slice(xd, [0, 0, 0, 0], [1, 1, 1, 64]), dim=-1))
    # embedding gather from a device index (uint32 row-major [1, n])
    table = ttnn.from_torch(torch.randn(4128, 2048).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    tablet = ttnn.from_torch(torch.randn(4128, 2048).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    idx = ttnn.from_torch(torch.tensor([[7]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    timed("embedding_rm_1", lambda: ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT))
    timed("embedding_tile_1", lambda: ttnn.embedding(idx, tablet, layout=ttnn.TILE_LAYOUT))
    idx8 = ttnn.from_torch(torch.arange(8, dtype=torch.int32).reshape(1, 8), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    timed("embedding_rm_8", lambda: ttnn.embedding(idx8, table, layout=ttnn.TILE_LAYOUT))
    # fp32 code (tile) -> uint32 row-major index
    code = ttnn.from_torch(torch.tensor([[[[5.0]]]]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)

    def to_idx():
        u = ttnn.typecast(code, ttnn.uint32)
        return ttnn.to_layout(ttnn.reshape(u, (1, 1)), ttnn.ROW_MAJOR_LAYOUT)

    o = timed("fp32_to_uint32_rm", to_idx)
    if o is not None:
        print("idx value", ttnn.to_torch(o), flush=True)
        o2 = ttnn.embedding(o, table, layout=ttnn.TILE_LAYOUT)
        print("gather ok", float((ttnn.to_torch(o2).float().reshape(-1)[:2048] - ttnn.to_torch(table).float()[5]).abs().max()), flush=True)
    # readback paths
    small = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    t0 = time.time()
    for _ in range(100):
        ttnn.to_torch(small)
    res["to_torch_small_us"] = round((time.time() - t0) / 100 * 1e6, 1)
    try:
        host = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        t0 = time.time()
        for _ in range(100):
            ttnn.copy_device_to_host_tensor(small, host, blocking=True)
        res["copy_d2h_blocking_us"] = round((time.time() - t0) / 100 * 1e6, 1)
        ttnn.copy_device_to_host_tensor(small, host, blocking=False)
        ttnn.synchronize_device(mesh)
        res["copy_d2h_nonblocking"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["copy_d2h"] = "ERR " + str(e).splitlines()[0][:200]
    try:
        ev = ttnn.record_event(mesh, 0)
        ttnn.event_synchronize(ev)
        res["events"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["events"] = "ERR " + str(e).splitlines()[0][:200]
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open("/opt/ep0/port7/bench_sample_ops.json", "w"), indent=1)
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
