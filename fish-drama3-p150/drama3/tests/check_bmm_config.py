# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: the grouped-MoE bmm program configs (prefill_grouped._bmm_config) vs the default program, every M
bucket, real layer-0 expert weights: max |diff|, PCC, worst row PCC, time."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.drama3.host.generate import MODEL, TT_ENV


def pcc(a, b):
    a, b = a.flatten().double() - a.double().mean(), b.flatten().double() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.tt.device_check import build_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.moe.operations import expert_compute_kernel_config
    from models.demos.blackhole.qwen36.tt.moe.prefill_grouped import M_BUCKETS, _bmm_config

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        W = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(MODEL))[0].feed_forward.experts.weights
        ckc = expert_compute_kernel_config()
        g = torch.Generator().manual_seed(0)
        for M in M_BUCKETS:
            for name, w, down in (("gate_up", W.gate_up_proj, False), ("down", W.down_proj, True)):
                E, K, N = w.shape[1], w.shape[-2], w.shape[-1]
                cfg = _bmm_config(mesh, M, K, N, down)
                if cfg is None:
                    print(json.dumps(dict(M=M, mm=name, cfg="default")), flush=True)
                    continue
                x = ttnn.from_torch(torch.randn(1, E, M, K, generator=g) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                ref = ttnn.to_torch(ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
                outs = []
                for rep in range(3):
                    outs.append(ttnn.to_torch(ttnn.matmul(x, w, program_config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float())
                o = outs[0]
                rows = o.reshape(-1, N)
                rr = ref.reshape(-1, N)
                worst = min(pcc(rows[i], rr[i]) for i in range(0, rows.shape[0], max(1, rows.shape[0] // 512)))
                print(json.dumps(dict(M=M, mm=name, pcm=cfg.per_core_M, ibw=cfg.in0_block_w, max_diff=float((o - ref).abs().max()),
                                      pcc=round(pcc(o, ref), 6), worst_row_pcc=round(worst, 5),
                                      repeat_identical=all(torch.equal(outs[0], z) for z in outs[1:]))), flush=True)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
