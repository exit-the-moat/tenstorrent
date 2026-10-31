# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track A: the grouped-MoE suffix prefill (qwen36 moe/prefill_grouped.py) stage by stage on random weights
of the real shapes / dtypes (gate_up BFP4 [1,256,2048,1024], down BFP8 [1,256,512,2048]), S = 128 rows of which T
are valid, M rows per expert. Synced per stage, median of reps.

    (interactive env) python -m models.demos.blackhole.drama3.tests.bench_grouped [--T 77] [--M 32]
"""
import argparse
import json
import os
import time

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=77)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import models.demos.blackhole.qwen36.tt.moe.prefill_grouped as pg
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu, expert_compute_kernel_config

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    S, H, E, I, K, T, M = 128, 2048, 256, 512, 8, a.T, a.M
    torch.manual_seed(0)
    gu = ttnn.from_torch(torch.randn(1, E, H, 2 * I) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    dp = ttnn.from_torch(torch.randn(1, E, I, H) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)

    class W:
        gate_up_proj, down_proj = gu, dp

    class C:
        num_experts, top_k = E, K

    # routing: top-8 of random logits with a few hot experts; keep the busiest at <= M
    while True:
        lg = torch.randn(T, E)
        lg[:, :4] += 1.0
        w, e = lg.softmax(-1).topk(K, -1)
        if int(torch.bincount(e.reshape(-1), minlength=E).max()) <= M:
            break
    r = torch.zeros(S, E)
    r[torch.arange(T).unsqueeze(1), e] = w / w.sum(-1, keepdim=True)
    xh = torch.randn(1, 1, S, H)
    xh[:, :, T:] = 0
    x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    rd = ttnn.from_torch(r.reshape(1, 1, S, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    res = dict(T=T, M=M, need=int(torch.bincount(e.reshape(-1), minlength=E).max()))

    def med(f, reps=a.reps):
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            o = f()
            ttnn.synchronize_device(mesh)
            ts.append(time.time() - t0)
            if o is not None:
                for t_ in (o if isinstance(o, (list, tuple)) else [o]):
                    ttnn.deallocate(t_)
        return round(1e3 * sorted(ts)[len(ts) // 2], 3)

    ref = ttnn.to_torch(pg.grouped_prefill_forward(x, rd, W, C, mesh, valid_len=T, force_M=M)).float()
    res["whole_ms"] = med(lambda: pg.grouped_prefill_forward(x, rd, W, C, mesh, valid_len=T, force_M=M))

    # stages of _grouped with the host lists prebuilt
    rr = ttnn.to_torch(rd).float().reshape(S, E)
    rows = torch.arange(T)
    w_, e_ = torch.topk(rr[rows], K, dim=-1)
    fe = e_.reshape(-1)
    order = torch.argsort(fe, stable=True)
    counts = torch.bincount(fe, minlength=E)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(fe)
    slot[order] = torch.arange(fe.numel()) - starts[fe[order]]
    dest = fe * M + slot
    gi_h = torch.zeros(E * M, dtype=torch.long)
    gi_h[dest] = rows.repeat_interleave(K)
    sw_h = torch.zeros(E * M)
    sw_h[dest] = w_.reshape(-1)
    bi_h = torch.full((S, K), E * M, dtype=torch.long)
    bi_h[rows] = dest.reshape(T, K)
    st = {}
    st["readback_ms"] = med(lambda: (ttnn.to_torch(rd), None)[1])
    st["upload3_ms"] = med(lambda: [pg._u32(mesh, gi_h), pg._u32(mesh, bi_h),
                                    ttnn.from_torch(sw_h.reshape(1, E, M, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)])
    gi = pg._u32(mesh, gi_h)
    bi = pg._u32(mesh, bi_h)
    sw = ttnn.from_torch(sw_h.reshape(1, E, M, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    x_rm = ttnn.to_layout(ttnn.reshape(x, (S, H)), ttnn.ROW_MAJOR_LAYOUT)
    st["x_to_rm_ms"] = med(lambda: ttnn.to_layout(ttnn.reshape(x, (S, H)), ttnn.ROW_MAJOR_LAYOUT))
    st["gather_ms"] = med(lambda: ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
    xg = ttnn.reshape(ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, E, M, H))
    ckc = expert_compute_kernel_config()
    kw = dict(core_grid=ttnn.CoreGrid(y=10, x=11)) if os.environ.get("QWEN36_GROUPED_SMALL_GRID") == "1" else {}
    st["mm_gu_ms"] = med(lambda: ttnn.matmul(xg, gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw))
    ug = ttnn.matmul(xg, gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw)
    st["swiglu_ms"] = med(lambda: apply_swiglu(ug))
    act = apply_swiglu(ug)
    st["mm_dn_ms"] = med(lambda: ttnn.matmul(act, dp, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw))
    dn = ttnn.matmul(act, dp, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw)
    st["mul_sw_ms"] = med(lambda: ttnn.multiply(dn, sw))
    zero = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    st["concat_ms"] = med(lambda: ttnn.concat([ttnn.reshape(dn, (1, 1, E * M, H)), zero], dim=2))
    table = ttnn.concat([ttnn.reshape(dn, (1, 1, E * M, H)), zero], dim=2)
    st["table_to_rm_ms"] = med(lambda: ttnn.to_layout(ttnn.reshape(table, (E * M + 32, H)), ttnn.ROW_MAJOR_LAYOUT))
    trm = ttnn.to_layout(ttnn.reshape(table, (E * M + 32, H)), ttnn.ROW_MAJOR_LAYOUT)
    st["back_gather_ms"] = med(lambda: ttnn.embedding(bi, trm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
    pk = ttnn.reshape(ttnn.embedding(bi, trm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, S, K, H))
    st["sum_ms"] = med(lambda: ttnn.sum(pk, dim=2))
    res["stages"] = st
    res["stages_sum_ms"] = round(sum(st.values()), 3)

    # variant: gather and combine as matmuls with host-built one-hot / weight matrices
    G = torch.zeros(E * M, S)
    G[dest, rows.repeat_interleave(K)] = 1.0
    Wc = torch.zeros(S, E * M)
    Wc[rows.repeat_interleave(K), dest] = w_.reshape(-1).to(torch.bfloat16).float()
    hifi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    v = {}
    v["upload_G_Wc_ms"] = med(lambda: [ttnn.from_torch(G, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
                                        ttnn.from_torch(Wc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)])
    Gd = ttnn.from_torch(G, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    Wd = ttnn.from_torch(Wc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    x2 = ttnn.reshape(x, (S, H))
    v["gather_mm_ms"] = med(lambda: ttnn.matmul(Gd, x2, compute_kernel_config=hifi, dtype=ttnn.bfloat16))
    dn2 = ttnn.reshape(dn, (E * M, H))
    v["combine_mm_ms"] = med(lambda: ttnn.matmul(Wd, dn2, compute_kernel_config=hifi, dtype=ttnn.bfloat16))
    res["matmul_variant"] = v
    xg2 = ttnn.to_torch(ttnn.matmul(Gd, x2, compute_kernel_config=hifi, dtype=ttnn.bfloat16)).float()
    xg1 = ttnn.to_torch(ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)).float().reshape(E * M, H)
    res["gather_mm_equal"] = bool(torch.equal(xg1, xg2))
    out2 = ttnn.to_torch(ttnn.matmul(Wd, dn2, compute_kernel_config=hifi, dtype=ttnn.bfloat16)).float().reshape(S, H)
    rf = ref.reshape(S, H)
    res["combine_mm_maxdiff"] = float((out2 - rf).abs().max())
    res["combine_mm_pcc"] = float(torch.corrcoef(torch.stack([out2[:T].flatten(), rf[:T].flatten()]))[0, 1])
    # variant B: one-hot gather / weighted combine matrices built on the card by scatter from [S, K] lists
    idx_h = torch.zeros(S, K, dtype=torch.long)
    idx_h[rows] = dest.reshape(T, K)
    one_h = torch.zeros(S, K)
    one_h[rows] = 1.0
    wk_h = torch.zeros(S, K)
    wk_h[rows] = w_.reshape(T, K)
    up16 = lambda t: ttnn.from_torch(t.reshape(1, 1, S, K).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
    upbf = lambda t: ttnn.from_torch(t.reshape(1, 1, S, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
    b = {}
    b["upload_lists_ms"] = med(lambda: [up16(idx_h), upbf(one_h), upbf(wk_h)])
    idx_d, one_d, wk_d = up16(idx_h), upbf(one_h), upbf(wk_h)
    zeros = ttnn.zeros([1, 1, S, E * M], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    b["scatter_ms"] = med(lambda: ttnn.scatter(zeros, dim=-1, index=idx_d, src=wk_d))
    Gt = ttnn.scatter(zeros, dim=-1, index=idx_d, src=one_d)
    Wb = ttnn.scatter(zeros, dim=-1, index=idx_d, src=wk_d)
    x4 = ttnn.reshape(x, (1, 1, S, H))
    b["gather_mmT_ms"] = med(lambda: ttnn.matmul(Gt, x4, transpose_a=True, compute_kernel_config=hifi, dtype=ttnn.bfloat16))
    dn4 = ttnn.reshape(dn, (1, 1, E * M, H))
    b["combine_mm_ms"] = med(lambda: ttnn.matmul(Wb, dn4, compute_kernel_config=hifi, dtype=ttnn.bfloat16))
    xgb = ttnn.to_torch(ttnn.matmul(Gt, x4, transpose_a=True, compute_kernel_config=hifi, dtype=ttnn.bfloat16)).float().reshape(E * M, H)
    real = torch.zeros(E * M, dtype=torch.bool)
    real[dest] = True
    b["gather_equal_on_real_rows"] = bool(torch.equal(xgb[real], xg1[real]))
    b["gather_zero_on_pad_rows"] = bool((xgb[~real] == 0).all())
    outb = ttnn.to_torch(ttnn.matmul(Wb, dn4, compute_kernel_config=hifi, dtype=ttnn.bfloat16)).float().reshape(S, H)
    b["combine_maxdiff"] = float((outb - rf).abs().max())
    b["combine_pcc"] = float(torch.corrcoef(torch.stack([outb[:T].flatten(), rf[:T].flatten()]))[0, 1])
    b["Wc_equal_host"] = bool(torch.equal(ttnn.to_torch(Wb).float().reshape(S, E * M), Wc))
    res["scatter_variant"] = b
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
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
