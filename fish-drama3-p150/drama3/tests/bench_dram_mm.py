# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: what DRAM bandwidth does a one-token matmul reach on the p150? Day 4b tuned 1D-mcast programs
(tt/decode_linear.py TABLE) vs DRAM-sharded matmuls (weight width-sharded over the 8 DRAM banks, activation
width-sharded in L1), random weights, one-op traces replayed N times.

    python -m models.demos.blackhole.drama3.tests.bench_dram_mm [--out f.json]
"""
import argparse
import json
import math
import time

import torch

import ttnn

SHAPES = [  # (K, N, dtype, what)
    (2048, 6144, "bf16", "tower w1 / w3"),
    (6144, 2048, "bf16", "tower w2"),
    (2048, 9216, "bf16", "tower wqkv"),
    (4096, 2048, "bf16", "tower wo"),
    (2048, 2048, "bf16", "tower eh half"),
    (2048, 6144, "bfp8", "tower w1 / w3 bfp8"),
    (6144, 2048, "bfp8", "tower w2 bfp8"),
    (2048, 12352, "bfp8", "GDN in-proj"),
    (4096, 2048, "bfp8", "GDN / attn o_proj"),
    (2048, 9216, "bfp8", "attn qkv merged"),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port7/bench_dram_mm.json")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    from models.demos.blackhole.drama3.tt import decode_linear

    grid = mesh.compute_with_storage_grid_size()
    dram = mesh.dram_grid_size()
    nb = dram.x
    print("grid", grid.x, grid.y, "dram banks", nb, flush=True)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    ckc2 = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
    res = []

    def timed(fn):
        fn()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(a.n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.time() - t0) / a.n * 1e6
        r = ttnn.to_torch(out).float()
        ttnn.release_trace(mesh, tid)
        return us, r

    # empty-trace floor: one tiny op
    xs = ttnn.from_torch(torch.randn(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    us0, _ = timed(lambda: ttnn.add(xs, xs))
    print("tiny add replay us", round(us0, 2), flush=True)
    res.append(dict(what="tiny add (replay floor)", us=round(us0, 2)))
    try:
        for K, N, dt, what in SHAPES:
            torch.manual_seed(0)
            wt = torch.randn(K, N) * 0.02
            xt = torch.randn(1, 1, 32, K)
            xt[:, :, 1:] = 0
            ref = (xt[0, 0, :1].double() @ wt.double()).float()
            mb = K * N * BYTES[dt] / 1e6
            row = dict(what=what, K=K, N=N, dtype=dt, MB=round(mb, 2), cands=[])
            w_il = ttnn.from_torch(wt, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=mesh)
            x_il = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for ck_name, ck in (("HiFi4", ckc), ("HiFi2", ckc2)):
                # 1) default / tuned interleaved
                cfg = decode_linear._config(mesh, w_il, None)
                kw = dict(compute_kernel_config=ck)
                if cfg is not None:
                    kw["program_config"] = cfg
                us, r = timed(lambda: ttnn.linear(x_il, w_il, **kw))
                err = float((r[0, 0, 0] - ref[0]).abs().max() / ref.abs().max())
                row["cands"].append(dict(kind="interleaved" + ("-tuned" if cfg else "-default"), ckc=ck_name, us=round(us, 2), GBps=round(mb / us * 1e3, 1), err=round(err, 5)))
                print(what, row["cands"][-1], flush=True)
            # 2) DRAM-sharded weight
            Np = math.ceil(N / (32 * nb)) * 32 * nb
            wpad = torch.zeros(K, Np)
            wpad[:, :N] = wt
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))}), (K, Np // nb), ttnn.ShardOrientation.ROW_MAJOR),
            )
            w_ds = ttnn.from_torch(wpad, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc)
            kt, nt = K // 32, Np // 32
            for nc in (8, 16, 24, 32, 40, 48, 64, 80, 96):
                if kt % nc or nt % nc or nc > grid.x * grid.y:
                    continue
                cols = min(grid.x, nc)
                rows_ = math.ceil(nc / cols)
                if cols * rows_ != nc:
                    # build an exact core set of nc cores row-major
                    pass
                cores = ttnn.num_cores_to_corerangeset(nc, grid, row_wise=True)
                in_mc = ttnn.create_sharded_memory_config((32, K // nc), cores, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
                out_mc = ttnn.create_sharded_memory_config((32, Np // nc), cores, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
                x_sh = ttnn.to_memory_config(x_il, in_mc)
                kpc = kt // nc
                for ibw in sorted({d for d in (1, 2, 4, 8, 16) if kpc % d == 0} | {kpc} if kpc <= 16 else {d for d in (1, 2, 4, 8, 16) if kpc % d == 0}):
                    for nw in (1, 2):
                        for ck_name, ck in (("HiFi4", ckc), ("HiFi2", ckc2)):
                            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                                in0_block_w=ibw, per_core_M=1, per_core_N=nt // nc, fused_activation=None, num_workers_per_dram_bank=nw
                            )
                            try:
                                us, r = timed(lambda: ttnn.linear(x_sh, w_ds, program_config=pc, memory_config=out_mc, compute_kernel_config=ck))
                            except Exception as e:  # noqa: BLE001
                                row["cands"].append(dict(kind="dram_sharded", cores=nc, ibw=ibw, nw=nw, ckc=ck_name, error=str(e).splitlines()[0][:160]))
                                continue
                            err = float((r[0, 0, 0, :N] - ref[0]).abs().max() / ref.abs().max())
                            row["cands"].append(dict(kind="dram_sharded", cores=nc, ibw=ibw, nw=nw, ckc=ck_name, us=round(us, 2), GBps=round(mb / us * 1e3, 1), err=round(err, 5)))
                            print(what, row["cands"][-1], flush=True)
                ttnn.deallocate(x_sh)
            ok = [c for c in row["cands"] if "us" in c]
            row["best"] = min(ok, key=lambda c: c["us"]) if ok else None
            print("BEST", what, row["best"], flush=True)
            res.append(row)
            ttnn.deallocate(w_ds)
            ttnn.deallocate(w_il)
            json.dump(res, open(a.out, "w"), indent=1)
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
