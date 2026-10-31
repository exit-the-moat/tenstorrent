# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port8: the one-program residual add + norm (tt/addnorm_fused.py) vs ttnn.add + decode_norm.ShardedRowNorm on
random rows (DRAM and L1 inputs, the residual's massive channel), plus traced device time of a chain of 40
add + norm boundaries each way (the residual of one boundary is the x of the next, as in the backbone).

    (fast env) python -m models.demos.blackhole.drama3.tests.check_addnorm [--out json]
"""
import argparse
import json
import os
import time

import torch

import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.tt.addnorm_fused import AddNormFused, AddNormFused2
    from models.demos.blackhole.drama3.tt.decode_norm import ShardedRowNorm

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 << 20)
    res = {"checks": [], "timing_us_per_boundary": {}}
    try:
        H = 2048
        ref = ShardedRowNorm(mesh, H)
        fz = (AddNormFused2 if os.environ.get("DRAMA3_ADDNORM_V", "2") == "2" else AddNormFused)(mesh, H)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
        torch.manual_seed(0)
        L1 = ttnn.L1_MEMORY_CONFIG
        for trial in range(8):
            xh = torch.randn(1, 1, H) * (0.3 + trial)
            dh = torch.randn(1, 1, H) * (0.1 + 0.5 * trial)
            if trial % 2:
                xh[0, 0, 1269] = 2800.0  # the backbone residual's massive channel
            wh = 1.0 + 0.1 * torch.randn(H)
            mx = L1 if trial % 3 == 0 else ttnn.DRAM_MEMORY_CONFIG
            md = L1 if trial % 4 == 1 else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.from_torch(xh.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mx)
            d = ttnn.from_torch(dh.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=md)
            w = ttnn.from_torch(wh.to(torch.bfloat16).reshape(1, 1, H // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            h_ref = ttnn.add(x, d)
            y_ref = ttnn.to_torch(ref(h_ref, w, 1e-6, ckc)).float().flatten()
            h_ref = ttnn.to_torch(h_ref).float().flatten()
            h_f, y_f = fz(x, d, w, 1e-6, out_mc=L1 if trial % 2 else None)
            h_f, y_f = ttnn.to_torch(h_f).float().flatten(), ttnn.to_torch(y_f).float().flatten()
            hd = ttnn.from_torch(h_ref.reshape(1, 1, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            _, y_n = fz(hd, None, w, 1e-6)
            y_n = ttnn.to_torch(y_n).float().flatten()
            hf = (xh.to(torch.bfloat16).float() + dh.to(torch.bfloat16).float()).flatten()
            hb = hf.to(torch.bfloat16).float()
            t = hb * torch.rsqrt((hb * hb).mean() + 1e-6) * wh.to(torch.bfloat16).float()
            r = dict(trial=trial, x_l1=mx is L1, d_l1=md is L1, h_bit_equal_ttnn_add=bool(torch.equal(h_f, h_ref)),
                     h_maxabs_vs_add=round(float((h_f - h_ref).abs().max()), 5), h_bit_equal_torch_bf16=bool(torch.equal(h_f, hb)),
                     y_pcc_vs_ref=round(pcc(y_f, y_ref), 7), y_maxabs_vs_ref=round(float((y_f - y_ref).abs().max()), 5),
                     y_pcc_ref_vs_torch=round(pcc(y_ref, t), 7), y_pcc_fused_vs_torch=round(pcc(y_f, t), 7),
                     norm_only_pcc_vs_torch=round(pcc(y_n, t), 7))
            res["checks"].append(r)
            print(json.dumps(r), flush=True)

        N = 40
        x0 = ttnn.from_torch(torch.randn(1, 1, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        d0 = ttnn.from_torch((0.01 * torch.randn(1, 1, H)).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        def chain_ref():
            x = x0
            for _ in range(N):
                h = ttnn.add(x, d0)
                y = ref(h, w, 1e-6, ckc, L1)
                ttnn.deallocate(y)
                if x is not x0:
                    ttnn.deallocate(x)
                x = h
            return x

        def chain_fused():
            x = x0
            for _ in range(N):
                h, y = fz(x, d0, w, 1e-6, out_mc=L1)
                ttnn.deallocate(y)
                if x is not x0:
                    ttnn.deallocate(x)
                x = h
            return x

        def chain_norm_ref():
            for _ in range(N):
                ttnn.deallocate(ref(x0, w, 1e-6, ckc, L1))
            return None

        def chain_norm_fused():
            for _ in range(N):
                ttnn.deallocate(fz(x0, None, w, 1e-6, out_mc=L1)[1])
            return None

        outs = {}
        for name, fn in (("add+sharded_norm", chain_ref), ("fused", chain_fused), ("sharded_norm_only", chain_norm_ref),
                         ("fused_norm_only", chain_norm_fused)):
            o = fn()
            if o is not None:
                ttnn.deallocate(o)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            o = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.time() - t0) / 100 / N * 1e6
            res["timing_us_per_boundary"][name] = round(us, 2)
            print(name, "traced chain of", N, round(us, 2), "us per boundary", flush=True)
            if o is not None:
                outs[name] = ttnn.to_torch(o).float().flatten()
            ttnn.release_trace(mesh, tid)
        res["chain_out_bit_equal"] = bool(torch.equal(outs["add+sharded_norm"], outs["fused"]))
        print("chain residual bit-equal", res["chain_out_bit_equal"], flush=True)
    finally:
        ttnn.close_device(mesh)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
