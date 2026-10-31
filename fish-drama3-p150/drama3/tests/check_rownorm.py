# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: the one-program row norm (tt/rownorm_fused.py) vs decode_norm.ShardedRowNorm on random rows (DRAM
and L1 inputs, a residual-like massive channel), plus traced device time of both.

    (fast env) python -m models.demos.blackhole.drama3.tests.check_rownorm
"""
import json
import time

import torch

import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    from models.demos.blackhole.drama3.tt.decode_norm import ShardedRowNorm
    from models.demos.blackhole.drama3.tt.rownorm_fused import RowNormFused

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    try:
        H = 2048
        ref, fz = ShardedRowNorm(mesh, H), RowNormFused(mesh, H)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
        torch.manual_seed(0)
        res = []
        for trial in range(6):
            xh = torch.randn(1, 1, H) * (0.3 + trial)
            if trial % 2:
                xh[0, 0, 1269] = 2800.0  # the backbone residual's massive channel
            wh = 1.0 + 0.1 * torch.randn(H)
            mc = ttnn.L1_MEMORY_CONFIG if trial % 3 == 0 else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.from_torch(xh.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc)
            w = ttnn.from_torch(wh.to(torch.bfloat16).reshape(1, 1, H // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            a = ttnn.to_torch(ref(x, w, 1e-6, ckc)).float().flatten()
            b = ttnn.to_torch(fz(x, w, 1e-6)).float().flatten()
            xf = xh.to(torch.bfloat16).float().flatten()
            t = xf * torch.rsqrt((xf * xf).mean() + 1e-6) * wh.to(torch.bfloat16).float()
            r = dict(trial=trial, in_l1=trial % 3 == 0, pcc_fused_vs_ref=round(pcc(b, a), 7), maxabs=round(float((a - b).abs().max()), 5),
                     pcc_ref_vs_torch=round(pcc(a, t), 7), pcc_fused_vs_torch=round(pcc(b, t), 7), bit_equal=bool(torch.equal(a, b)))
            res.append(r)
            print(json.dumps(r), flush=True)
        x = ttnn.from_torch(torch.randn(1, 1, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        for name, fn in (("sharded3", lambda: ref(x, w, 1e-6, ckc)), ("fused", lambda: fz(x, w, 1e-6))):
            fn()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(200):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(name, "traced", round((time.time() - t0) / 200 * 1e6, 2), "us", flush=True)
            ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
