# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: op-level device time of qwen36's routed-expert decode (moe/decode.py) on layer 0, each op traced alone
and replayed 100 times.

    python -m models.demos.blackhole.drama3.tests.profile_moe_decode
"""
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)


def timed(mesh, fn, n=100):
    out = fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ttnn.release_trace(mesh, tid)
    return out, (time.time() - t0) / n * 1e3


def main():
    from models.demos.blackhole.drama3.tt.device_check import build_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=64 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        layer = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(args.CKPT_DIR))[0]
        moe = layer.feed_forward
        W, E, I, H = moe.experts.weights, 256, 512, 2048
        x = ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        res = {}
        r, res["router"] = timed(mesh, lambda: moe.router(x))
        sp, res["sparsity_to_row_major"] = timed(mesh, lambda: ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT))
        ug, res["gate_up_sparse_matmul"] = timed(
            mesh,
            lambda: ttnn.sparse_matmul(
                x, W.gate_up_proj, sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]),
                program_config=_build_sparse_matmul_config(1, 2 * I), dtype=ttnn.bfloat16,
            ),
        )
        ug4, res["reshape_permute_1"] = timed(mesh, lambda: ttnn.permute(ttnn.reshape(ug, (1, E, 1, 2 * I)), (2, 0, 1, 3)))
        di, res["swiglu"] = timed(mesh, lambda: apply_swiglu(ug4))
        di4, res["reshape_transpose_2"] = timed(
            mesh, lambda: ttnn.reshape(ttnn.transpose(ttnn.reshape(di, (1, E, I)), 1, 0), (1, E, 1, I))
        )
        dn, res["down_sparse_matmul"] = timed(
            mesh,
            lambda: ttnn.sparse_matmul(
                di4, W.down_proj, sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]),
                program_config=_build_sparse_matmul_config(1, H), is_input_a_sparse=True, dtype=ttnn.bfloat16,
            ),
        )
        ns, res["permute_3"] = timed(mesh, lambda: ttnn.reshape(ttnn.permute(dn, (0, 2, 1, 3)), (1, E, H)))
        r3 = ttnn.reshape(r, (1, E, 1))
        nm, res["routing_mul"] = timed(mesh, lambda: ttnn.mul(ns, r3))
        _, res["sum_over_experts"] = timed(mesh, lambda: ttnn.sum(nm, dim=1))
        for k_, v_ in res.items():
            print(f"{k_:28s} {v_:.3f} ms", flush=True)
        print("total", round(sum(res.values()), 3), "ms", flush=True)
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
