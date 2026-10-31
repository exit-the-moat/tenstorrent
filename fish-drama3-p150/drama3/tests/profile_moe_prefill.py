# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: op-level timing of qwen36's routed-expert prefill chunk (moe/prefill.py::_process_prefill_chunk) on
layer 0 of Drama 3, for chunk lengths 128 / 512. Random hidden states; real router.

    python -m models.demos.blackhole.drama3.tests.profile_moe_prefill
"""
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)


def main():
    from models.demos.blackhole.drama3.tt.device_check import build_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.moe import prefill as P
    from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu, expert_compute_kernel_config

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        layer = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(args.CKPT_DIR))[0]
        moe = layer.feed_forward
        W, cfg = moe.experts.weights, moe.config
        for chunk in (128, 512):
            torch.manual_seed(0)
            x = ttnn.from_torch(torch.randn(1, 1, chunk, 2048) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            r = moe.router(x)
            E, I, H = 256, W.intermediate_size_per_device, 2048
            tm = {}

            def tick(name, t0):
                ttnn.synchronize_device(mesh)
                tm[name] = tm.get(name, 0) + time.time() - t0
                return time.time()

            for it in range(3):  # 1 warm + 2 timed
                if it == 1:
                    tm = {}
                ttnn.synchronize_device(mesh)
                t = time.time()
                g = chunk // 32
                hg = ttnn.reshape(x, (1, g, 32, H))
                rt = ttnn.reshape(r, (1, g, 32, E))
                tmask = ttnn.max(rt, dim=2, keepdim=True)
                tmask = ttnn.to_layout(tmask, ttnn.ROW_MAJOR_LAYOUT)
                sp = ttnn.reshape(tmask, (1, 1, g, E))
                t = tick("sparsity", t)
                ug = ttnn.sparse_matmul(
                    hg, W.gate_up_proj, sparsity=sp, nnz=None, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(32, 2 * I),
                    compute_kernel_config=expert_compute_kernel_config(), dtype=ttnn.bfloat16,
                )
                t = tick("gate_up_sparse_matmul", t)
                ug = ttnn.transpose(ug, 1, 3)
                ug = ttnn.reshape(ug, (1, E, chunk, 2 * I))
                t = tick("transpose_reshape", t)
                di = apply_swiglu(ug)
                ug.deallocate(True)
                di = ttnn.reshape(di, (1, E, chunk, I))
                t = tick("swiglu", t)
                dn = ttnn.sparse_matmul(
                    di, W.down_proj, sparsity=moe.experts.prefill_sparsity, nnz=E, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(chunk, H),
                    compute_kernel_config=expert_compute_kernel_config(), is_input_a_sparse=True, dtype=ttnn.bfloat16,
                )
                di.deallocate(True)
                t = tick("down_sparse_matmul(all experts)", t)
                ns = ttnn.reshape(dn, (1, E, chunk, H))
                rp = ttnn.permute(r, (0, 3, 2, 1))
                ns = ttnn.mul(ns, rp)
                t = tick("routing_mul", t)
                ns = ttnn.experimental.fast_reduce_nc(ns, dims=[1])
                t = tick("reduce_over_experts", t)
                for tt_ in (ns, rp, dn, sp, tmask):
                    ttnn.deallocate(tt_)
            print(json.dumps({"chunk": chunk, **{k: round(v / 2 * 1e3, 1) for k, v in tm.items()}, "unit": "ms per call"}), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
