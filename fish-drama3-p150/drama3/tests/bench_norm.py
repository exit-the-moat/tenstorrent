# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: one-token RMSNorm [1,1,1,2048] on a P150: interleaved (as the decode layers run it) vs width-sharded
(with and without the interleaved<->sharded conversions); traced, device time per call, and max |diff|."""
import json
import time

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    try:
        H = 2048
        x_t = torch.randn(1, 1, 1, H) * 3
        w_t = torch.rand(H) + 0.5
        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        w = ttnn.from_torch(w_t.reshape(1, 1, H // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False)
        ref = ttnn.to_torch(ttnn.rms_norm(x, epsilon=1e-6, weight=w, compute_kernel_config=ckc)).float()

        def timed(fn, n=200):
            ttnn.deallocate(fn())
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
            return ms, float((r - ref).abs().max())

        res = {"interleaved": timed(lambda: ttnn.rms_norm(x, epsilon=1e-6, weight=w, compute_kernel_config=ckc))}
        for n_cores, grid in ((8, (8, 1)), (16, (8, 2)), (32, (8, 4))):
            shard_w = H // n_cores
            mc = ttnn.create_sharded_memory_config(
                shape=(32, shard_w), core_grid=ttnn.CoreGrid(y=grid[1], x=grid[0]), strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
            )
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid, subblock_w=min(4, shard_w // 32), block_h=1, block_w=shard_w // 32, inplace=False,
            )
            xs = ttnn.to_memory_config(x, mc)

            def sharded_only(xs=xs, pc=pc, mc=mc):
                return ttnn.rms_norm(xs, epsilon=1e-6, weight=w, program_config=pc, memory_config=mc, compute_kernel_config=ckc)

            def round_trip(pc=pc, mc=mc):
                a = ttnn.to_memory_config(x, mc)
                y = ttnn.rms_norm(a, epsilon=1e-6, weight=w, program_config=pc, memory_config=mc, compute_kernel_config=ckc)
                return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

            try:
                res[f"sharded{n_cores}_only"] = timed(sharded_only)
                res[f"sharded{n_cores}_roundtrip"] = timed(round_trip)
            except Exception as e:  # noqa: BLE001
                res[f"sharded{n_cores}"] = str(e)[:150]
        print(json.dumps(res), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
