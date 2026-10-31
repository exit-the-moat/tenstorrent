# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: grouped-MoE batched expert matmul at small / large M: default program vs core_grid-picked programs and
the bmm program config; time + determinism (5 repeats bit-exact) + max |diff| vs default."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.drama3.host.generate import MODEL, TT_ENV


def main():
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.tt.device_check import build_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.moe.operations import expert_compute_kernel_config

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        W = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(MODEL))[0].feed_forward.experts.weights
        ckc = expert_compute_kernel_config()
        g = torch.Generator().manual_seed(0)
        for M in (32, 64, 640):
            for name, w in (("gate_up", W.gate_up_proj), ("down", W.down_proj)):
                E, K = w.shape[1], w.shape[-2]
                x = ttnn.from_torch(torch.randn(1, E, M, K, generator=g) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                ref = ttnn.to_torch(ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
                variants = {"default": {}}
                for gy, gx in ((10, 11), (8, 8), (4, 8)):
                    variants[f"grid{gx}x{gy}"] = dict(core_grid=ttnn.CoreGrid(y=gy, x=gx))
                for vn, kw in variants.items():
                    try:
                        outs = []
                        for _ in range(5):
                            outs.append(ttnn.to_torch(ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw)).float())
                        det = all(torch.equal(outs[0], o) for o in outs[1:])
                        ttnn.synchronize_device(mesh)
                        t0 = time.time()
                        for _ in range(10):
                            ttnn.deallocate(ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw))
                        ttnn.synchronize_device(mesh)
                        ms = (time.time() - t0) / 10 * 1e3
                        print(json.dumps(dict(M=M, mm=name, variant=vn, ms=round(ms, 2), deterministic=det,
                                              max_diff=float((outs[0] - ref).abs().max()))), flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(json.dumps(dict(M=M, mm=name, variant=vn, error=str(e)[:120])), flush=True)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
