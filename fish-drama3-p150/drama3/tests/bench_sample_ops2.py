# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: fp32 top-k / sort at small widths, gathers of fp32 values by a device index list, cumsum forms."""
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
            res[name] = "ERR " + str(e).splitlines()[0][:300]
            print(name, res[name], flush=True)
            return None

    torch.manual_seed(0)
    for w in (32, 64, 128, 256, 1024):
        x = ttnn.from_torch(torch.randn(1, 1, 1, w), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        for k in (32, 64):
            if k <= w:
                timed(f"topk_fp32_w{w}_k{k}", lambda: ttnn.topk(x, k, dim=-1, largest=True, sorted=True))
        timed(f"sort_fp32_w{w}", lambda: ttnn.sort(x, dim=-1, descending=True))
    # gather fp32 values at 64 indices of a 4128 row
    src = torch.randn(1, 1, 1, 4128)
    xs = ttnn.from_torch(src, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    ids = torch.randperm(4128)[:64].reshape(1, 1, 1, 64)
    idt = ttnn.from_torch(ids.int(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
    idt16 = ttnn.from_torch(ids.int(), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh)
    for name, it in (("u32", idt), ("u16", idt16)):
        o = timed(f"gather_{name}", lambda: ttnn.gather(xs, -1, index=it))
        if o is not None:
            res[f"gather_{name}_err"] = float((ttnn.to_torch(o).float().reshape(-1)[:64] - src.reshape(-1)[ids.reshape(-1)]).abs().max())
    # bf16 topk indices -> gather fp32 (the index dtype topk returns)
    xb = ttnn.typecast(xs, ttnn.bfloat16)
    v, i = ttnn.topk(xb, 64, dim=-1, largest=True, sorted=True)
    print("topk index dtype", i.dtype, i.layout, flush=True)
    o = timed("gather_topk_idx", lambda: ttnn.gather(xs, -1, index=i))
    if o is not None:
        res["gather_topk_idx_err"] = float((ttnn.to_torch(o).float().reshape(-1)[:64] - src.reshape(-1)[ttnn.to_torch(i).long().reshape(-1)[:64]]).abs().max())
    # cumsum forms on a 64-wide fp32 row: ttnn.cumsum vs a matmul with an upper-triangular ones matrix (fp32 acc)
    r = ttnn.from_torch(torch.rand(1, 1, 1, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    tri = ttnn.from_torch(torch.triu(torch.ones(64, 64)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    ck = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    timed("cumsum_fp32_64", lambda: ttnn.cumsum(r, dim=-1))
    o = timed("cumsum_matmul_64", lambda: ttnn.matmul(r, tri, compute_kernel_config=ck, dtype=ttnn.float32))
    if o is not None:
        rr = ttnn.to_torch(r).float().reshape(-1)[:64]
        res["cumsum_matmul_err"] = float((ttnn.to_torch(o).float().reshape(-1)[:64] - torch.cumsum(rr, 0)).abs().max())
    json.dump(res, open("/opt/ep0/port7/bench_sample_ops2.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
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
