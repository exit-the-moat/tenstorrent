# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 prefetch prototype: can the DRAM-core tensor prefetcher (DRISC kernels on the DRAM cores stream the
next matmul's weights into a global CB in worker L1) hide the tower's small ops behind weight streaming?

One tower layer = 6 one-token matmuls with the tower's shapes / dtypes (eh_h, q|gate|k|v, o_proj, w1, w3, w2),
random weights, with stand-in small ops (rms_norm / add on the activation row) between them where the tower has
its norms, attention and SiLU * mul. Everything for --layers layers is one trace, replayed --n times.

    (a)  today's style: interleaved DRAM weights, tuned 1D-mcast configs of tt/decode_linear.py
    (a') same cores / program config as (b), interleaved weights, no prefetcher (bit-exact reference for (b))
    (b)  weights receiver-contiguous in DRAM, streamed by the tensor prefetcher into one global CB on the
         matmul cores (mcast_in0 1D matmul reading in1 from the GCB); the per-frame request is captured
         into the trace (capture_into_trace=True)
Each variant with and without the small ops; the small ops alone too.

    python -m models.demos.blackhole.drama3.tests.bench_tower_prefetch [--layers 1] [--n 50] [--gcb_kb 512]
"""
import argparse
import json
import math
import os
import time

import torch

import ttnn

T = 32
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
TILE_B = {"bf16": 2048, "bfp8": 1088}
# (name, K, N, dtype, stand-in small ops after it); live tower: one-token matmuls (port8 prof_s8.json)
MATS = [
    ("eh_h", 2048, 2048, "bf16", "addnorm"),  # -> fused add + norm (6-8 us)
    ("qkvg", 2048, 9216, "bfp8", "attn"),  # -> attention pre 12.3 + 2 KV updates 8 + SDPA 27.9 + post 4 us
    ("o_proj", 4096, 2048, "bfp8", "addnorm"),
    ("w1", 2048, 6144, "bf16", None),
    ("w3", 2048, 6144, "bf16", "silumul"),  # -> SiLU 4.3 + multiply 5.3 us
    ("w2", 6144, 2048, "bf16", "addnorm"),
]
# stand-in ops per gap kind on a [1, 32, 2048] row: n = rms_norm (~19 us here), a = add (~2.4 us + ~1.4 us dispatch);
# default sized to the measured tower gaps (add+norm ~7 us, attention ~52 us, SiLU * mul ~9.6 us)
GAP_DEFAULT = "addnorm=aa,attn=naaaaaaaa,silumul=aa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gcb_kb", type=int, default=512, help="global CB size per receiver core, KB")
    ap.add_argument("--rpb", type=int, default=4, help="receivers per DRAM bank (ring = banks x rpb)")
    ap.add_argument("--ibw", type=int, default=0, help="in0_block_w of the a2 / b matmuls (in1 page = ibw x per_core_N tiles); 0: per matmul from --page_kb")
    ap.add_argument("--page_kb", type=int, default=64, help="--ibw 0: largest in0_block_w whose in1 page fits this")
    ap.add_argument("--req", default="capture", help="capture: the frame's prefetch request is captured into the trace; ahead: all frames queued up front")
    ap.add_argument("--variants", default="c,a,a2,b", help="comma list of c (small ops), a, a2 (a'), b")
    ap.add_argument("--fid", default="HiFi4")
    ap.add_argument("--silu", type=int, default=0, help="1: SiLU fused into w1 (as the tower does)")
    ap.add_argument("--gap", default=GAP_DEFAULT, help="stand-in ops per gap kind (n = rms_norm, a = add)")
    ap.add_argument("--gap_rep", type=int, default=1, help="repeat every gap this many times (run-ahead probe)")
    ap.add_argument("--prof", default="", help="device-profile only this timed variant (run with TT_METAL_DEVICE_PROFILER=1)")
    ap.add_argument("--out", default="/opt/ep0/port10/prefetch/bench.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 << 20)
    res = dict(args=vars(a))
    try:
        run(mesh, a, res)
    finally:
        json.dump(res, open(a.out, "w"), indent=1)
        ttnn.close_mesh_device(mesh)
    if a.prof:
        from models.demos.blackhole.drama3.tests.profile_ops_dev import parse_log

        ops = parse_log(1)[0]
        res["prof"] = dict(
            name=a.prof, span_us=round(ops[-1][1] + ops[-1][2], 1), sum_kernel_us=round(sum(o[2] for o in ops), 1),
            ops=[dict(i=i, start_us=round(st, 2), dur_us=round(d, 2), risc=rk) for i, (rid, st, d, rk) in enumerate(ops)],
        )
        for o in res["prof"]["ops"]:
            print("prof", o, flush=True)
        print("prof span", res["prof"]["span_us"], "sum kernels", res["prof"]["sum_kernel_us"], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


def run(mesh, a, res):
    from models.demos.blackhole.drama3.tt import decode_linear

    grid = mesh.compute_with_storage_grid_size()
    nb = mesh.dram_grid_size().x
    sup = ttnn.experimental.is_tensor_prefetcher_supported(mesh)
    print("grid", grid.x, grid.y, "dram banks", nb, "tensor prefetcher supported", sup, flush=True)
    res.update(grid=[grid.x, grid.y], dram_banks=nb, prefetcher_supported=bool(sup))
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, a.fid), fp32_dest_acc_en=True, packer_l1_acc=False
    )
    L = a.layers
    ring = nb * a.rpb
    rows = a.rpb
    variants = a.variants.split(",")

    # ---- inputs (one fixed activation row per K) and the small-op row ----
    torch.manual_seed(1)
    x_t = {}
    x_il = {}
    for K in sorted({m[1] for m in MATS}):
        t = torch.zeros(1, 1, T, K)
        t[:, :, 0] = torch.randn(K)
        x_t[K] = t
        x_il[K] = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    hrow = ttnn.from_torch(torch.randn(1, 1, T, 2048) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    gap_ops = {k: v for k, v in (kv.split("=") for kv in a.gap.split(","))}

    def gap(kind):
        if kind is None:
            return
        y = hrow
        for c in gap_ops[kind] * a.gap_rep:
            y = ttnn.rms_norm(y, epsilon=1e-6, compute_kernel_config=ckc) if c == "n" else ttnn.add(y, hrow)

    # ---- weights: host copies per layer (bf16-rounded), device copies per variant ----
    t0 = time.time()
    w_host = []
    for l in range(L):
        g = torch.Generator().manual_seed(100 + l)
        w_host.append([(torch.randn(K, N, generator=g) * 0.02).to(torch.bfloat16) for _, K, N, _, _ in MATS])
    print(f"host weights {time.time() - t0:.1f} s", flush=True)
    nbytes_layer = sum(K * N * TILE_B[d] / 1024 for _, K, N, d, _ in MATS)
    res["weight_MB_per_layer"] = round(nbytes_layer / 1e6, 2)

    def timed(name, fn, n=a.n, pre=None):
        """capture fn() as one trace, replay n times; us per replay. --prof: only that variant, 2 replays.
        pre(frames): called first with the number of frames fn's matmuls will run (eager + replays)."""
        if a.prof and name != a.prof:
            return None
        if a.prof:
            n = 0
        if pre is not None:
            pre(n + 2 + (1 if a.prof else 0))
        fn()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # one untimed replay
        if a.prof:
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
        t1 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t1) / max(n, 1) * 1e6
        ttnn.release_trace(mesh, tid)
        if a.prof:
            return out
        r = dict(us_frame=round(us, 1), us_layer=round(us / L, 1), GBps=round(nbytes_layer * L / us / 1e3, 1))
        print(f"[{name}] {r}", flush=True)
        res.setdefault("timing", {})[name] = r
        return out

    # ---- (c) small ops alone ----
    if "c" in variants:
        timed("c_small_ops", lambda: [gap(m[4]) for _ in range(L) for m in MATS])

    # ---- (a) today's style ----
    outs = {}
    if "a" in variants:
        wa = [[ttnn.from_torch(w, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=mesh) for w, (_, _, _, d, _) in zip(wl, MATS)] for wl in w_host]
        cfg = {i: decode_linear._config(mesh, wa[0][i], None) for i in range(len(MATS))}
        print("decode_linear configs", {MATS[i][0]: (str(c) if c is None else (c.per_core_N, c.in0_block_w, c.compute_with_storage_grid_size.x, c.compute_with_storage_grid_size.y)) for i, c in cfg.items()}, flush=True)

        def frame_a(gaps):
            o = []
            for l in range(L):
                for i, (_, K, N, d, g) in enumerate(MATS):
                    kw = dict(program_config=cfg[i]) if cfg[i] is not None else {}
                    if a.silu and MATS[i][0] == "w1":
                        kw = dict(activation="silu")
                    o.append(ttnn.linear(x_il[K], wa[l][i], compute_kernel_config=ckc, **kw))
                    if gaps:
                        gap(g)
            return o

        timed("a_mm_only", lambda: frame_a(False))
        outs["a"] = timed("a_with_small_ops", lambda: frame_a(True))
        for wl in wa:
            for w in wl:
                ttnn.deallocate(w)

    # ---- (a') and (b): same cores / program config; N padded to a multiple of ring tiles (zeros) ----
    npad = [math.ceil(N / T / ring) * ring * T for _, K, N, _, _ in MATS]
    res["N_padded"] = {m[0]: npad[i] for i, m in enumerate(MATS) if npad[i] != m[2]}
    w_pad = [[torch.nn.functional.pad(w, (0, npad[i] - w.shape[1])) for i, w in enumerate(wl)] for wl in w_host]

    def pcfg(K, N, d, act=None):
        kt, nt = K // T, N // T
        pcn = nt // ring
        ibw = a.ibw or max(b for b in (32, 16, 8, 4, 2, 1) if kt % b == 0 and (b * pcn * TILE_B[d] <= a.page_kb * 1024 or b == 1) and 2 * b * pcn * TILE_B[d] <= a.gcb_kb * 1024)
        assert nt % ring == 0 and kt % ibw == 0, (K, N, ring, ibw)
        sw = max(s for s in (4, 3, 2, 1) if pcn % s == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(nb, rows),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sw,
            out_block_h=1,
            out_block_w=pcn,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=rows,
            untilize_out=False,
            stream_in1=False,
        )

    silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    pcs = [pcfg(K, npad[i], d, silu if (a.silu and nm == "w1") else None) for i, (nm, K, N, d, _) in enumerate(MATS)]

    if "a2" in variants:
        wa = [[ttnn.from_torch(w, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=mesh) for w, (_, _, _, d, _) in zip(wl, MATS)] for wl in w_pad]

        def frame_a2(gaps):
            o = []
            for l in range(L):
                for i, (_, K, N, d, g) in enumerate(MATS):
                    o.append(ttnn.linear(x_il[K], wa[l][i], program_config=pcs[i], compute_kernel_config=ckc))
                    if gaps:
                        gap(g)
            return o

        timed("a2_mm_only", lambda: frame_a2(False))
        outs["a2"] = timed("a2_with_small_ops", lambda: frame_a2(True))
        for wl in wa:
            for w in wl:
                ttnn.deallocate(w)

    if "b" in variants:
        if not sup:
            res["b_error"] = "tensor prefetcher not supported on this device"
            print(res["b_error"], flush=True)
        else:
            run_b(mesh, a, res, outs, x_il, w_pad, pcs, ckc, gap, timed, nb, ring, rows)

    # ---- correctness ----
    def tt(o):
        return ttnn.to_torch(o).float()[0, 0, 0, :N]

    cmp = {}
    for l in range(L) if L <= 2 else (0, L - 1):
        for i, (nm, K, N, d, _) in enumerate(MATS):
            k = l * len(MATS) + i
            ref = (x_t[K][0, 0, 0].double() @ w_host[l][i].double()).float()
            if a.silu and nm == "w1":
                ref = torch.nn.functional.silu(ref)
            row = {}
            for v in ("a", "a2", "b"):
                if outs.get(v) is not None:
                    y = tt(outs[v][k])
                    row[f"{v}_pcc_torch"] = round(float(torch.corrcoef(torch.stack([y, ref]))[0, 1]), 6)
            if outs.get("b") is not None and outs.get("a2") is not None:
                yb, ya = tt(outs["b"][k]), tt(outs["a2"][k])
                row["b_vs_a2_bitexact"] = bool(torch.equal(yb, ya))
                row["b_vs_a2_maxabs"] = float((yb - ya).abs().max())
            if outs.get("b") is not None and outs.get("a") is not None:
                yb, ya = tt(outs["b"][k]), tt(outs["a"][k])
                row["b_vs_a_pcc"] = round(float(torch.corrcoef(torch.stack([yb, ya]))[0, 1]), 6)
                row["b_vs_a_maxabs"] = float((yb - ya).abs().max())
            cmp[f"L{l}_{nm}"] = row
            print("check", l, nm, row, flush=True)
    res["check"] = cmp


def run_b(mesh, a, res, outs, x_il, w_host, pcs, ckc, gap, timed, nb, ring, rows):
    L = a.layers
    dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})

    def recv_contig(w, d):  # [K, N] -> ND-sharded, ring shards of [K, N / ring], shard s on bank s % nb
        K, N = w.shape
        spec = ttnn.NdShardSpec(ttnn.Shape([K, N // ring]), dram, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
        return ttnn.from_torch(w.reshape(1, 1, K, N), dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))

    t0 = time.time()
    wb = [[recv_contig(w, d) for w, (_, _, _, d, _) in zip(wl, MATS)] for wl in w_host]
    print(f"recv-contig weights {time.time() - t0:.1f} s", flush=True)
    # bank b -> ring positions b, b + nb, ... = column b of the (nb x rows) receiver grid (strided, ROUND_ROBIN_1D)
    b2r = [(b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, s), ttnn.CoreCoord(b, s)) for s in range(rows)])) for b in range(nb)]
    size = a.gcb_kb * 1024
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(mesh, pcs, wb[0], b2r, size)
    bcs = [ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(pcs[i], wb[0][i], gcb) for i in range(len(MATS))]
    pages = [pcs[i].in0_block_w * pcs[i].per_core_N * TILE_B[MATS[i][3]] for i in range(len(MATS))]
    res["b_setup"] = dict(
        ring=ring, receiver_grid=[nb, rows], gcb_bytes_per_core=size, block_counts=dict(zip([m[0] for m in MATS], bcs)),
        page_bytes=dict(zip([m[0] for m in MATS], pages)),
        per_core_N=dict(zip([m[0] for m in MATS], [p.per_core_N for p in pcs])),
    )
    print("b setup", res["b_setup"], flush=True)
    req = [(wb[l][i], bcs[i]) for l in range(L) for i in range(len(MATS))]

    def frame_b(gaps, capture):
        if a.req == "capture":
            ttnn.experimental.queue_tensor_prefetcher_request(mesh, req, global_cb=gcb, capture_into_trace=capture)
        o = []
        for l in range(L):
            for i, (_, K, N, d, g) in enumerate(MATS):
                o.append(ttnn.linear(x_il[K], wb[l][i], program_config=pcs[i], compute_kernel_config=ckc, global_cb=gcb))
                if gaps:
                    gap(g)
        return o

    ttnn.experimental.start_tensor_prefetcher(mesh)
    try:
        # timed(): fn() once eagerly (request sent now), then captured (request held in the trace)
        state = {"cap": False}

        def fb(gaps):
            def f():
                r = frame_b(gaps, state["cap"])
                state["cap"] = True
                return r
            return f

        def ahead(frames):  # weights are constant: queue every frame's stream before running any of them
            ttnn.experimental.queue_tensor_prefetcher_request(mesh, req * frames, global_cb=gcb)

        pre = ahead if a.req == "ahead" else None
        state["cap"] = False
        timed("b_mm_only", fb(False), pre=pre)
        state["cap"] = False
        outs["b"] = timed("b_with_small_ops", fb(True), pre=pre)
    finally:
        ttnn.experimental.stop_tensor_prefetcher(mesh)
        ttnn.synchronize_device(mesh)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
