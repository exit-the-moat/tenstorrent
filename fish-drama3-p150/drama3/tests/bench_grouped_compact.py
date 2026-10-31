# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7 TTFA: the grouped-MoE prefill matmuls over all E = 256 experts (today: batched matmul on the 11x10 grid)
vs only the A active ones (suffix prefill: 146 of 256 on average, port7/need.json) via sparse_matmul's gather mode
(compact [1, A, M, K] input, expert ids as indices). Random bfp4 gate_up / bfp8 down weights; one-op traces."""
import json
import os
import time

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config

    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False)
    E, H, I = 256, 2048, 512
    torch.manual_seed(0)
    gu_t = torch.randn(1, E, H, 2 * I) * 0.02
    dp_t = torch.randn(1, E, I, H) * 0.02
    gu = ttnn.from_torch(gu_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    dp = ttnn.from_torch(dp_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    res = {}

    def timed(name, fn, n=20):
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
            ms = (time.time() - t0) / n * 1e3
            r = ttnn.to_torch(out).float()
            ttnn.release_trace(mesh, tid)
            res[name] = round(ms, 3)
            print(name, round(ms, 3), "ms", flush=True)
            return r
        except Exception as e:  # noqa: BLE001
            res[name] = "ERR " + str(e).splitlines()[0][:300]
            print(name, res[name], flush=True)
            return None

    for M in (32, 64):
        xa = torch.randn(1, E, M, H) * 0.5
        xd = ttnn.from_torch(xa, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        g = ttnn.CoreGrid(y=10, x=11)
        full = timed(f"bmm_gate_up_E256_M{M}", lambda: ttnn.matmul(xd, gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=g))
        ad = ttnn.from_torch(torch.randn(1, E, M, I) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        timed(f"bmm_down_E256_M{M}", lambda: ttnn.matmul(ad, dp, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=g))
        for A in (128, 160, 192):
            ids = torch.randperm(E)[:A].sort().values
            idx = ttnn.from_torch(ids.reshape(1, A).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            sp = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            xc = ttnn.from_torch(xa[:, ids].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for ibw in (8, 32):
                cfg = _build_sparse_matmul_config(M, 2 * I, in0_block_w=ibw)
                r = timed(f"sparse_gate_up_A{A}_M{M}_ibw{ibw}", lambda: ttnn.sparse_matmul(
                    xc, gu, sparsity=sp, indices=idx, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]),
                    program_config=cfg, compute_kernel_config=ckc, is_input_a_sparse=True, dtype=ttnn.bfloat16))
                if r is not None and full is not None:
                    a = r.reshape(A, M, 2 * I)
                    b = full.reshape(E, M, 2 * I)[ids]
                    d = (a - b).abs().max() / b.abs().max()
                    res[f"sparse_gate_up_A{A}_M{M}_ibw{ibw}_relerr_vs_bmm"] = float(d)
                    print("  shape", list(r.shape), "relerr vs bmm", float(d), flush=True)
            ac = ttnn.from_torch(torch.randn(1, A, M, I) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for ibw in (4, 16):
                cfg = _build_sparse_matmul_config(M, H, in0_block_w=ibw)
                timed(f"sparse_down_A{A}_M{M}_ibw{ibw}", lambda: ttnn.sparse_matmul(
                    ac, dp, sparsity=sp, indices=idx, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]),
                    program_config=cfg, compute_kernel_config=ckc, is_input_a_sparse=True, dtype=ttnn.bfloat16))
        json.dump(res, open("/opt/ep0/port7/bench_grouped_compact.json", "w"), indent=1)
    ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
