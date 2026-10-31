# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: the tower prefill's 128-row matmuls (bf16 / bfp8 weights, HiFi4 + fp32 acc) with ttnn's default program vs
2D-mcast program configs, one-op traces. TTFA: the tower prefill of the suffix bucket is ~43 ms for 13 layers."""
import json
import math
import time

import torch

import ttnn

SHAPES = [(2048, 2048, "bf16"), (2048, 9216, "bfp8"), (4096, 2048, "bfp8"), (2048, 6144, "bf16"), (6144, 2048, "bf16")]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    grid = mesh.compute_with_storage_grid_size()
    res = {}

    def timed(fn, n=50):
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
        r = ttnn.to_torch(out).float()
        ttnn.release_trace(mesh, tid)
        return us, r

    M = 128
    for K, N, dt in SHAPES:
        torch.manual_seed(0)
        w = torch.randn(K, N) * 0.02
        x = torch.randn(1, M, K)
        ref = (x.double() @ w.double()).float()
        wd = ttnn.from_torch(w, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=mesh)
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        key = f"{K}x{N}_{dt}"
        us, r = timed(lambda: ttnn.linear(xd, wd, compute_kernel_config=ckc))
        best = dict(kind="default", us=round(us, 1), err=float((r - ref).abs().max() / ref.abs().max()))
        row = dict(default=best, cands=[])
        mt, kt, nt = M // 32, K // 32, N // 32
        for gy in (1, 2, 4):
            if mt % gy:
                continue
            for gx in range(2, grid.x + 1):
                pn = math.ceil(nt / gx)
                for ibw in (1, 2, 4, 8):
                    if kt % ibw:
                        continue
                    pm = mt // gy
                    sw = max(d for d in (1, 2, 4) if pn % d == 0)
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=sw,
                        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True,
                    )
                    try:
                        us, r = timed(lambda: ttnn.linear(xd, wd, program_config=pc, compute_kernel_config=ckc))
                    except Exception:  # noqa: BLE001
                        continue
                    err = float((r - ref).abs().max() / ref.abs().max())
                    c = dict(gx=gx, gy=gy, ibw=ibw, pn=pn, us=round(us, 1), err=round(err, 5))
                    row["cands"].append(c)
                    if us < best["us"] and err < 2 * row["default"]["err"] + 1e-3:
                        best = dict(c, kind="2d")
        row["best"] = best
        res[key] = row
        print(key, "default", row["default"], "best", best, flush=True)
        json.dump(res, open("/opt/ep0/port7/bench_prefill_mm.json", "w"), indent=1)
    ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
