# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: program-config sweep for the grouped MoE prefill's batched expert matmuls [1,E,M,H] x [1,E,H,2I] and
[1,E,M,I] x [1,E,I,H] at small M (default config: 7.7 ms gate_up at M = 32, ~0.6 ms of weight bandwidth).

    (fast env) python -m models.demos.blackhole.drama3.tests.sweep_grouped_mm
"""
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
        layer = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(MODEL))[0]
        W = layer.feed_forward.experts.weights
        E = layer.feed_forward.experts.config.num_experts
        H = args.dim
        print("weights", W.gate_up_proj.shape, W.gate_up_proj.dtype, W.down_proj.shape, W.down_proj.dtype, flush=True)
        grid = mesh.compute_with_storage_grid_size()
        print("grid", grid.x, grid.y, flush=True)
        ckc = expert_compute_kernel_config()

        def timed(fn, n=5):
            out = fn()
            ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(n):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            return (time.time() - t0) / n * 1e3

        for M in (64, 128, 640):
            for name, w in (("gate_up", W.gate_up_proj), ("down", W.down_proj)):
                K, N = w.shape[-2], w.shape[-1]
                x = ttnn.from_torch(torch.randn(1, E, M, K) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                ref = ttnn.to_torch(ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
                r = dict(M=M, mm=name, default_ms=round(timed(lambda: ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)), 2))
                print(json.dumps(r), flush=True)
                Mt, Kt, Nt = M // 32, K // 32, N // 32
                best = None
                for pcm, pcn, ibw, sw in [(m_, Nt, b, 4) for m_ in (1, 2, 4) if Mt % m_ == 0 for b in (2, 4, 8)]:
                    if True:
                        if Kt % ibw:
                            continue
                        sh = 1
                        cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                            compute_with_storage_grid_size=(grid.x, grid.y),
                            in0_block_w=ibw,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            per_core_M=pcm,
                            per_core_N=pcn,
                        )
                        try:
                            ms = timed(lambda: ttnn.matmul(x, w, program_config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
                            out = ttnn.matmul(x, w, program_config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
                            d = float((ttnn.to_torch(out).float() - ref).abs().max())
                            ttnn.deallocate(out)
                        except Exception as e:  # noqa: BLE001
                            print(json.dumps(dict(M=M, mm=name, pcn=pcn, ibw=ibw, err=str(e)[:120])), flush=True)
                            continue
                        rr = dict(M=M, mm=name, per_core_M=pcm, per_core_N=pcn, in0_block_w=ibw, sub_w=sw, ms=round(ms, 2), max_diff=d)
                        print(json.dumps(rr), flush=True)
                        if best is None or ms < best["ms"]:
                            best = rr
                print("BEST", json.dumps(best), flush=True)
                ttnn.deallocate(x)
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
