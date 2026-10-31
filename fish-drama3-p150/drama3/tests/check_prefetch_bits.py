# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 tower prefetch: are the prefetched 8x8 matmuls bit-identical to today's decode_linear matmuls, per
in0_block_w, with the tower's real compute configs (eh_h / MLP: tower ckc HiFi4 fp32 acc, no packer L1 acc;
q|gate|k|v / o_proj: the attention decode ckc LoFi fp32 acc WITH packer L1 acc, where the K-block split could
change the fp32 summation order)? Random weights, one row; one GCB for every variant, one request in run order.

    python -m models.demos.blackhole.drama3.tests.check_prefetch_bits [--gcb_kb 512]
"""
import argparse
import json
import math
import os

import torch

import ttnn

T = 32
TILE_B = {"bf16": 2048, "bfp8": 1088}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
# (name, K, N, dtype, ckc kind, silu, in0_block_w candidates); first candidate = decode_linear's
MATS = [
    ("eh_h", 2048, 2048, "bf16", "tower", False, (4, 32, 64)),
    ("qkvg", 2048, 9216, "bfp8", "attn", False, (4, 8, 16)),
    ("o_proj", 4096, 2048, "bfp8", "attn", False, (16, 32, 64)),
    ("w1", 2048, 6144, "bfp8", "tower", True, (4, 16, 32)),
    ("w3", 2048, 6144, "bfp8", "tower", False, (4, 16, 32)),
    ("w2", 6144, 2048, "bfp8", "tower", False, (8, 64, 96)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcb_kb", type=int, default=512)
    ap.add_argument("--out", default="/opt/ep0/port10/prefetch/bits.json")
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {}
    try:
        run(mesh, a, res)
    finally:
        json.dump(res, open(a.out, "w"), indent=1)
        ttnn.close_mesh_device(mesh)
    print("DONE", flush=True)


def run(mesh, a, res):
    from models.demos.blackhole.drama3.tt import decode_linear

    decode_linear.install(mesh)
    ckcs = dict(
        tower=ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False),
        attn=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True),
    )
    nb = mesh.dram_grid_size().x
    ring, rows = 64, 8
    dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    torch.manual_seed(3)
    ref, wnd, cfgs, runs = {}, {}, [], []
    xs = {}
    for nm, K, N, d, kind, silu, ibws in MATS:
        if K not in xs:
            xs[K] = ttnn.from_torch(torch.randn(1, 1, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        w = (torch.randn(K, N) * 0.02).to(torch.bfloat16)
        wi = ttnn.from_torch(w, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=mesh)
        kw = dict(activation="silu") if silu else {}
        ref[nm] = ttnn.to_torch(ttnn.linear(xs[K], wi, compute_kernel_config=ckcs[kind], memory_config=ttnn.L1_MEMORY_CONFIG, **kw)).float().reshape(-1)[:N]
        # the prefetch copy: from the device weight's values (as the tower does), N padded to ring tiles
        wt = ttnn.to_torch(wi).float().reshape(K, N)
        npad = math.ceil(N / T / ring) * ring * T
        wt = torch.nn.functional.pad(wt, (0, npad - N))
        spec = ttnn.NdShardSpec(ttnn.Shape([K, npad // ring]), dram, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
        wnd[nm] = ttnn.from_torch(wt.reshape(1, 1, K, npad), dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))
        pcn = npad // T // ring
        for ibw in ibws:
            sw = max(s for s in (4, 3, 2, 1) if pcn % s == 0)
            c = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(nb, rows), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=sw,
                out_block_h=1, out_block_w=pcn, per_core_M=1, per_core_N=pcn, fuse_batch=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if silu else None, mcast_in0=True,
                gather_in0=False, hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=rows, untilize_out=False, stream_in1=False,
            )
            cfgs.append(c)
            runs.append((nm, K, N, kind, ibw, c))
    b2r = [(b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, s), ttnn.CoreCoord(b, s)) for s in range(rows)])) for b in range(nb)]
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(mesh, cfgs, [wnd[r[0]] for r in runs], b2r, a.gcb_kb * 1024)
    req = [(wnd[nm], ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(c, wnd[nm], gcb)) for nm, _, _, _, _, c in runs]
    ttnn.experimental.start_tensor_prefetcher(mesh)
    try:
        ttnn.experimental.queue_tensor_prefetcher_request(mesh, req, global_cb=gcb)
        outs = []
        for nm, K, N, kind, ibw, c in runs:
            outs.append(ttnn.linear(xs[K], wnd[nm], program_config=c, compute_kernel_config=ckcs[kind], global_cb=gcb, memory_config=ttnn.L1_MEMORY_CONFIG))
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.experimental.stop_tensor_prefetcher(mesh)
        ttnn.synchronize_device(mesh)
    for (nm, K, N, kind, ibw, c), o in zip(runs, outs):
        y = ttnn.to_torch(o).float().reshape(-1)
        tail = float(y[N:].abs().max()) if y.numel() > N else 0.0
        y = y[:N]
        r = dict(bitexact=bool(torch.equal(y, ref[nm])), maxabs=float((y - ref[nm]).abs().max()), ndiff=int((y != ref[nm]).sum()), pad_maxabs=tail,
                 page_kb=round(ibw * c.per_core_N * TILE_B[dict((m[0], m[3]) for m in MATS)[nm]] / 1024, 1))
        res[f"{nm}_ibw{ibw}"] = r
        print("bits", nm, ibw, r, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
