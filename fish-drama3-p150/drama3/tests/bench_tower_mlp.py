# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: the tower's one-token MLP gate/up as today (w1 + silu, w3, mul: 3 programs) vs [w3 | w1] as one
[2048, 12288] bf16 matmul + the SwiGLU kernel (2 programs), HiFi4 + fp32 accumulation as the tower; program
config sweep of the merged matmul. One-op... traces of the whole piece, replayed N times (8.6 us replay floor)."""
import json
import math
import time

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    from models.demos.blackhole.drama3.tt import decode_linear
    from models.demos.blackhole.drama3.tt.moe_fused_decode import Swiglu

    decode_linear.install(mesh)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    grid = mesh.compute_with_storage_grid_size()
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
            r = ttnn.to_torch(out).float()
            ttnn.release_trace(mesh, tid)
            res[name] = round(us, 2)
            print(name, round(us, 2), flush=True)
            return r
        except Exception as e:  # noqa: BLE001
            res[name] = "ERR " + str(e).splitlines()[0][:200]
            print(name, res[name], flush=True)
            return None

    torch.manual_seed(0)
    H, I = 2048, 6144
    w1, w3 = torch.randn(I, H) * 0.02, torch.randn(I, H) * 0.02
    x = torch.randn(1, 1, H)
    ref = (torch.nn.functional.silu(x.double() @ w1.double().T) * (x.double() @ w3.double().T)).float().reshape(-1)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    d1 = ttnn.from_torch(w1.T.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    d3 = ttnn.from_torch(w3.T.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    def today():
        u = ttnn.linear(xd, d1, activation="silu", compute_kernel_config=ckc)
        v = ttnn.linear(xd, d3, compute_kernel_config=ckc)
        return ttnn.mul(u, v)

    r = timed("today_w1silu_w3_mul", today)
    res["today_err"] = float((r.reshape(-1)[:I] - ref).abs().max() / ref.abs().max())
    gu = torch.cat([w3.T, w1.T], dim=-1).contiguous().to(torch.bfloat16)  # [H, 2I]: [up | gate]
    dgu = ttnn.from_torch(gu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    nt = 2 * I // 32
    for ncores in (64, 96, 48):
        sw = Swiglu(mesh, ncores)
        for pn in (4, 6, 8, 12):
            cores = math.ceil(nt / pn)
            if cores > grid.x * grid.y:
                continue
            for ibw in (2, 4, 8):
                sub = max(d for d in (1, 2, 4) if pn % d == 0)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(min(grid.x, cores), math.ceil(cores / grid.x)), in0_block_w=ibw, out_subblock_h=1,
                    out_subblock_w=sub, per_core_M=1, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True,
                )

                def merged(pc=pc, sw=sw):
                    g = ttnn.linear(xd, dgu, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
                    return sw(ttnn.reshape(g, (1, 1, 1, 2 * I)))

                r = timed(f"merged_sw{ncores}_pn{pn}_ibw{ibw}", merged)
                if r is not None:
                    res[f"merged_sw{ncores}_pn{pn}_ibw{ibw}_err"] = float((r.reshape(-1)[:I] - ref).abs().max() / ref.abs().max())
            if ncores != 64:
                break
    json.dump(res, open("/opt/ep0/port7/bench_tower_mlp.json", "w"), indent=1)
    ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
