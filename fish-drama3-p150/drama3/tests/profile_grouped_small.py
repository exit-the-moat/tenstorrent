# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: the grouped MoE prefill's pieces at small M (a 128-row suffix bucket), one layer, device sync per piece.

    (fast env) python -m models.demos.blackhole.drama3.tests.profile_grouped_small
"""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV


def main():
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.tt.device_check import build_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.drama3.host.generate import MODEL
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu, expert_compute_kernel_config

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        layer = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(MODEL))[0]
        moe = layer.feed_forward
        W, cfg = moe.experts.weights, moe.experts.config
        S, H, E = 128, args.dim, cfg.num_experts
        x = ttnn.from_torch(torch.randn(1, 1, S, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ckc = expert_compute_kernel_config()

        def sync():
            ttnn.synchronize_device(mesh)
            return time.time()

        for M in (32, 64, 128):
            for rep in range(3):
                res = {}
                t = sync()
                r = moe.router(x)
                t1 = sync(); res["router"] = t1 - t
                rh = ttnn.to_torch(r).float().reshape(S, E)
                t2 = time.time(); res["routing_readback"] = t2 - t1
                gi = ttnn.from_torch(torch.randint(0, S, (1, E * M)).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                x_rm = ttnn.to_layout(ttnn.reshape(x, (S, H)), ttnn.ROW_MAJOR_LAYOUT)
                t3 = sync(); res["upload_idx+to_rm"] = t3 - t2
                xg = ttnn.reshape(ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, E, M, H))
                t4 = sync(); res["gather"] = t4 - t3
                ug = ttnn.matmul(xg, W.gate_up_proj, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
                t5 = sync(); res["gate_up_mm"] = t5 - t4
                act = apply_swiglu(ug)
                t6 = sync(); res["swiglu"] = t6 - t5
                dn = ttnn.matmul(act, W.down_proj, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
                t7 = sync(); res["down_mm"] = t7 - t6
                sw = ttnn.from_torch(torch.rand(1, E, M, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                dn2 = ttnn.multiply(dn, sw)
                t8 = sync(); res["weight_mul"] = t8 - t7
                zero = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                table = ttnn.concat([ttnn.reshape(dn2, (1, 1, E * M, H)), zero], dim=2)
                table = ttnn.to_layout(ttnn.reshape(table, (E * M + 32, H)), ttnn.ROW_MAJOR_LAYOUT)
                t9 = sync(); res["concat+to_rm"] = t9 - t8
                bi = ttnn.from_torch(torch.randint(0, E * M, (1, S * 8)).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                pk = ttnn.reshape(ttnn.embedding(bi, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, S, 8, H))
                out = ttnn.sum(pk, dim=2)
                t10 = sync(); res["gather_back+sum"] = t10 - t9
                res["total"] = t10 - t
                if rep == 2:
                    print(json.dumps({"M": M, **{k: round(v * 1e3, 2) for k, v in res.items()}}), flush=True)
                for tt_ in (r, gi, x_rm, xg, ug, act, dn, sw, dn2, zero, table, bi, pk, out):
                    ttnn.deallocate(tt_)
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
