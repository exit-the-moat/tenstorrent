# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped routed-expert prefill for a SINGLE device (opt-in, Qwen36Experts.grouped_prefill = True).

The tile-sparse prefill (prefill.py) streams an expert's weights once per 32-token tile that touches it, and runs
down_proj for all experts on every row; for a few hundred tokens on one card that is ~0.3 s per layer. Here the
tokens are grouped by expert instead, so each expert's weights are read once:

  host: routing [S, E] -> per-expert token lists (top-k nonzeros of the first valid_len rows), padded to M rows
  device: gather the rows -> [1, E, M, H]; up|gate = x @ gate_up_proj (batched over E); swiglu; @ down_proj
          -> [1, E, M, H]; times each slot's routing weight; gather every token's k slots back and sum.
Same per-token math as the tile-sparse path (one expert MLP per (token, expert) pair, weighted sum over the top-k).
Rows past valid_len (bucket padding) come out as zeros. M is bucketed (M_BUCKETS) so no prompt compiles new programs.
"""

import os

import torch

import ttnn

from .operations import apply_swiglu, expert_compute_kernel_config

TILE = 32
# Allowed per-expert row counts M. M shapes every op here, so a free M would compile new programs whenever a new
# prompt routes differently; with a fixed set, warm_buckets() compiles them all up front (needed before a trace is
# parked). A need above the largest bucket splits the rows into halves and sums the parts. The small buckets serve
# short prefills (a line suffix after a cached prefix: ~100 rows, 11..72 rows on the busiest expert): the cost
# scales with E * M (22 ms per layer at M = 128 on a P150).
M_BUCKETS = (32, 64, 96, 128, 256, 384, 512, 640, 768)
# QWEN36_GROUPED_COMPACT=1 (port7, opt-in): at M = 32 with the matmul glue, only the experts the rows route to are
# computed (a line suffix uses ~146 of 256 per layer), as a compact [1, A, M, H] block through sparse_matmul's
# gather mode (expert ids as indices); A is bucketed. tests/bench_grouped_compact.py: gate_up + down 2.60 ms
# (batched matmul over 256) -> 1.48 / 1.82 / 2.19 ms at A = 128 / 160 / 192; at M = 64 the gather mode is slower.
A_BUCKETS = (64, 96, 128, 160, 192, 224)


def _bmm_config(mesh_device, M, K, N, down):
    """Batched expert matmul [1,E,M,K] x [1,E,K,N] (one expert per core block, full N per core): the default
    program re-reads the weights per 32-row block and runs at ~37 GB/s (7.6 ms for gate_up at M = 32 on a P150).
    Measured (tests/sweep_grouped_mm.py, ms gate_up / down): M=32 1.36 / 1.26, 64 1.44 / 1.31, 128 1.66 / 2.57,
    640 8.1 / default 9.98 (the full-N output block of down caps per_core_M at 2 in L1). None = the default.
    OFF by default (QWEN36_GROUPED_BMM_CONFIG=1 to opt in): each matmul matches the default program in isolation
    (tests/check_bmm_config.py, PCC 0.999997 on random inputs), but inside the Drama 3 backbone prefill the final
    norm drops to PCC 0.84 vs the CPU golden (0.994 with the default) and a layer goes NaN; not understood yet."""
    if os.environ.get("QWEN36_GROUPED_BMM_CONFIG", "0") != "1":
        return None
    if M < int(os.environ.get("QWEN36_GROUPED_BMM_MIN_M", "0")):
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if down:
        if Mt <= 2:
            pcm, ibw = Mt, 4
        elif Mt == 4:
            pcm, ibw = 2, 2
        else:
            return None
    else:
        pcm, ibw = next(m for m in (4, 3, 2, 1) if Mt % m == 0), 8
    ibw = int(os.environ.get("QWEN36_GROUPED_BMM_IBW", ibw))
    g = mesh_device.compute_with_storage_grid_size()
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y),
        in0_block_w=min(ibw, Kt),
        out_subblock_h=1,
        out_subblock_w=4 if Nt % 4 == 0 else 1,
        per_core_M=pcm,
        per_core_N=Nt,
    )


def _u32(mesh_device, t):
    return ttnn.from_torch(t.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)


_ZEROS = {}


def _zeros(mesh_device, S, N):
    """A persistent zero [1, 1, S, N] bf16 tile tensor per shape (scatter is out of place: it stays zero)."""
    if (S, N) not in _ZEROS:
        _ZEROS[(S, N)] = ttnn.zeros([1, 1, S, N], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    return _ZEROS[(S, N)]


def _glue_ckc():
    # fp32 accumulation, HiFi4: the one-hot gather is exact, the combine sums the k slots in fp32
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)


def grouped_prefill_forward(hidden_states, routing_weights, weights, config, mesh_device, valid_len=None, force_M=None, force_A=None):
    """hidden_states [1,1,S,H], routing_weights [1,1,S,E] (dense top-k weights) -> [1,1,S,H]."""
    S, E = hidden_states.shape[-2], config.num_experts
    n = S if valid_len is None else int(valid_len)
    r = ttnn.to_torch(routing_weights).float().reshape(S, E)
    return _grouped(hidden_states, r, torch.arange(n), weights, config, mesh_device, force_M, force_A)


def warm_buckets(experts, hidden_states, routing_weights):
    """Compile the grouped path for every M bucket (few valid rows, so every bucket fits)."""
    for m in M_BUCKETS:
        out = grouped_prefill_forward(
            hidden_states, routing_weights, experts.weights, experts.config, experts.mesh_device, valid_len=TILE, force_M=m
        )
        ttnn.deallocate(out)
    if _compact_on(hidden_states.shape[-2]):
        for a in A_BUCKETS:
            out = grouped_prefill_forward(
                hidden_states, routing_weights, experts.weights, experts.config, experts.mesh_device, valid_len=a // 8, force_M=TILE, force_A=a
            )
            ttnn.deallocate(out)


def _compact_on(S):
    return os.environ.get("QWEN36_GROUPED_COMPACT", "0") == "1" and os.environ.get("QWEN36_GROUPED_MM_GLUE", "0") == "1" and S <= 128


def grouped_prefill_forward_topk(hidden_states, top_k_values, top_k_indices, weights, config, mesh_device, valid_len=None):
    """QWEN36_GROUPED_TOPK_READ=1 (port6, opt-in): the same grouped prefill from the router's on-card top-k
    (values [1,1,S,k] normalised, ids [1,1,S,k]) instead of the scattered dense routing: the host reads S * k
    values and ids per layer instead of S * E routing weights and runs no top-k of its own."""
    S, K = hidden_states.shape[-2], config.top_k
    n = S if valid_len is None else int(valid_len)
    w = ttnn.to_torch(top_k_values).float().reshape(S, K)[:n]
    e = ttnn.to_torch(top_k_indices).to(torch.int64).reshape(S, K)[:n]
    return _grouped_we(hidden_states, w, e, torch.arange(n), weights, config, mesh_device)


def _grouped(hidden_states, r_all, rows, weights, config, mesh_device, force_M=None, force_A=None):
    """Expert MLPs of the given token rows (others -> 0)."""
    w, e = torch.topk(r_all[rows], config.top_k, dim=-1)  # [n, K] each token's top-k experts and weights
    return _grouped_we(hidden_states, w, e, rows, weights, config, mesh_device, force_M, force_A)


def _grouped_we(hidden_states, w, e, rows, weights, config, mesh_device, force_M=None, force_A=None):
    """Expert MLPs of the given token rows (others -> 0) from their top-k weights / experts w, e [n, K]."""
    S, H = hidden_states.shape[-2], hidden_states.shape[-1]
    E, K = config.num_experts, config.top_k
    n = rows.numel()
    flat_e = e.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e, minlength=E)
    need = int(counts.max())
    if force_M is not None:
        assert need <= force_M
        M = force_M
    elif need > M_BUCKETS[-1]:
        half = n // 2
        a = _grouped_we(hidden_states, w[:half], e[:half], rows[:half], weights, config, mesh_device)
        b = _grouped_we(hidden_states, w[half:], e[half:], rows[half:], weights, config, mesh_device)
        out = ttnn.add(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out
    else:
        M = next(m for m in M_BUCKETS if m >= need)
    active = torch.nonzero(counts).reshape(-1)
    if M == TILE and _compact_on(S) and (force_A is not None or active.numel() <= A_BUCKETS[-1]):
        A = force_A or next(a for a in A_BUCKETS if a >= active.numel())
        return _compact(hidden_states, w, e, rows, weights, config, mesh_device, active, A)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(flat_e)
    slot[order] = torch.arange(flat_e.numel()) - starts[flat_e[order]]
    dest = flat_e * M + slot  # [n*K] row of each (token, k) in the grouped [E*M] layout
    gather_idx = torch.zeros(E * M, dtype=torch.long)  # padding slots read token 0, weight 0
    gather_idx[dest] = rows.repeat_interleave(K)
    slot_w = torch.zeros(E * M)
    slot_w[dest] = w.reshape(-1)
    back_idx = torch.full((S, K), E * M, dtype=torch.long)  # other rows read the zero row E*M
    back_idx[rows] = dest.reshape(n, K)

    # QWEN36_GROUPED_MM_GLUE=1 (opt-in, port6): the row gather and the weighted k-slot combine as two matmuls with
    # [S, E*M] one-hot / routing-weight matrices that the card builds by scatter from [S, K] lists (one small upload)
    # instead of embedding gathers over row-major copies (bench_grouped.py: glue 2.85 -> 1.28 ms per layer at M = 32)
    mm_glue = os.environ.get("QWEN36_GROUPED_MM_GLUE", "0") == "1" and S <= 128 and E * M <= 65536
    if mm_glue:
        idx_h = torch.zeros(S, K, dtype=torch.int32)
        idx_h[rows] = dest.reshape(n, K).to(torch.int32)
        one_h = torch.zeros(S, K)
        one_h[rows] = 1.0
        wk_h = torch.zeros(S, K)
        wk_h[rows] = w.reshape(n, K)
        idx_d = ttnn.from_torch(idx_h.reshape(1, 1, S, K), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        one_d = ttnn.from_torch(one_h.reshape(1, 1, S, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        wk_d = ttnn.from_torch(wk_h.reshape(1, 1, S, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        zeros = _zeros(mesh_device, S, E * M)
        g_t = ttnn.scatter(zeros, dim=-1, index=idx_d, src=one_d)  # [1, 1, S, E*M]: 1 at (token, its slot row)
        w_c = ttnn.scatter(zeros, dim=-1, index=idx_d, src=wk_d)  # the same pattern with the routing weights
        for t_ in (idx_d, one_d, wk_d):
            ttnn.deallocate(t_)
        hifi = _glue_ckc()
        xg = ttnn.matmul(g_t, hidden_states, transpose_a=True, compute_kernel_config=hifi, dtype=ttnn.bfloat16)  # [1,1,E*M,H]
        ttnn.deallocate(g_t)
    else:
        x_rm = ttnn.to_layout(ttnn.reshape(hidden_states, (S, H)), ttnn.ROW_MAJOR_LAYOUT)
        gi = _u32(mesh_device, gather_idx)
        xg = ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, E*M, H]
        ttnn.deallocate(gi)
        ttnn.deallocate(x_rm)
    xg = ttnn.reshape(xg, (1, E, M, H))
    ckc = expert_compute_kernel_config()
    gu, dp = weights.gate_up_proj, weights.down_proj
    cfg_gu = _bmm_config(mesh_device, M, gu.shape[-2], gu.shape[-1], down=False)
    cfg_dn = _bmm_config(mesh_device, M, dp.shape[-2], dp.shape[-1], down=True)
    # QWEN36_GROUPED_GU_GRID_MIN_M=256 (opt-in): gate_up on an 8x8 core grid from this M up: the program ttnn
    # then picks is bit-identical to the default one at M = 640 and 1.8x faster (16.8 -> 9.2 ms, sweep_grouped_grid.py)
    gu_kw = {}
    min_m = int(os.environ.get("QWEN36_GROUPED_GU_GRID_MIN_M", "0"))
    if cfg_gu is None and min_m and M >= min_m:
        gu_kw["core_grid"] = ttnn.CoreGrid(y=8, x=8)
    # QWEN36_GROUPED_SMALL_GRID=1 (opt-in): below M = 256 both matmuls on the full 11x10 grid (M = 32: gate_up
    # 7.6 -> 1.4 ms, down 2.4 -> 1.3 ms; NOT bit-identical to the default: max |diff| 0.125 / 0.03)
    dn_kw = {}
    if os.environ.get("QWEN36_GROUPED_SMALL_GRID", "0") == "1" and M < 256:
        if cfg_gu is None:
            gu_kw["core_grid"] = ttnn.CoreGrid(y=10, x=11)
        dn_kw["core_grid"] = ttnn.CoreGrid(y=10, x=11)
    ug = ttnn.matmul(xg, gu, program_config=cfg_gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **gu_kw)  # [1, E, M, 2I]
    ttnn.deallocate(xg)
    act = apply_swiglu(ug)  # [up | gate] -> up * silu(gate)
    ttnn.deallocate(ug)
    if cfg_dn is not None:
        dn_kw = {}
    dn = ttnn.matmul(act, dp, program_config=cfg_dn, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **dn_kw)  # [1, E, M, H]
    ttnn.deallocate(act)
    if mm_glue:
        out = ttnn.matmul(w_c, ttnn.reshape(dn, (1, 1, E * M, H)), compute_kernel_config=hifi, dtype=ttnn.bfloat16)
        ttnn.deallocate(w_c)
        ttnn.deallocate(dn)
        return out  # [1, 1, S, H]; rows past valid_len are zero
    sw = ttnn.from_torch(slot_w.reshape(1, E, M, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    dn = ttnn.multiply(dn, sw)
    ttnn.deallocate(sw)
    # one zero row at E*M for the padded tokens, then gather each token's K slots and sum them
    zero = ttnn.from_torch(torch.zeros(1, 1, TILE, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    table = ttnn.concat([ttnn.reshape(dn, (1, 1, E * M, H)), zero], dim=2)
    ttnn.deallocate(dn)
    table = ttnn.to_layout(ttnn.reshape(table, (E * M + TILE, H)), ttnn.ROW_MAJOR_LAYOUT)
    bi = _u32(mesh_device, back_idx)
    per_k = ttnn.embedding(bi, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, S*K, H]
    ttnn.deallocate(bi)
    ttnn.deallocate(table)
    per_k = ttnn.reshape(per_k, (1, S, K, H))
    out = ttnn.sum(per_k, dim=2)  # [1, S, H]
    ttnn.deallocate(per_k)
    return ttnn.reshape(out, (1, 1, S, H))


def _compact(hidden_states, w, e, rows, weights, config, mesh_device, active, A):
    """QWEN36_GROUPED_COMPACT: the M = 32 grouped prefill over the active experts only (compact slot c = position
    in `active`, padded to A with the first active expert on zero rows), sparse_matmul gather mode, matmul glue."""
    from .decode import _build_sparse_matmul_config

    S, H = hidden_states.shape[-2], hidden_states.shape[-1]
    E, K, M = config.num_experts, config.top_k, TILE
    n = rows.numel()
    flat_e = e.reshape(-1)
    pos = torch.full((E,), -1, dtype=torch.long)
    pos[active] = torch.arange(active.numel())
    flat_c = pos[flat_e]
    order = torch.argsort(flat_c, stable=True)
    counts = torch.bincount(flat_c, minlength=A)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(flat_c)
    slot[order] = torch.arange(flat_c.numel()) - starts[flat_c[order]]
    dest = flat_c * M + slot  # row of each (token, k) in the compact [A*M] layout
    ids = torch.full((A,), int(active[0]) if active.numel() else 0, dtype=torch.int32)
    ids[: active.numel()] = active.to(torch.int32)
    idx_h = torch.zeros(S, K, dtype=torch.int32)
    idx_h[rows] = dest.reshape(n, K).to(torch.int32)
    one_h = torch.zeros(S, K)
    one_h[rows] = 1.0
    wk_h = torch.zeros(S, K)
    wk_h[rows] = w.reshape(n, K)
    idx_d = ttnn.from_torch(idx_h.reshape(1, 1, S, K), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    one_d = ttnn.from_torch(one_h.reshape(1, 1, S, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    wk_d = ttnn.from_torch(wk_h.reshape(1, 1, S, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    ex = ttnn.from_torch(ids.reshape(1, A), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    if ("sp", E) not in _ZEROS:  # the required (unread in gather mode) sparsity operand
        _ZEROS[("sp", E)] = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    sp = _ZEROS[("sp", E)]
    zeros = _zeros(mesh_device, S, A * M)
    g_t = ttnn.scatter(zeros, dim=-1, index=idx_d, src=one_d)  # [1, 1, S, A*M]
    w_c = ttnn.scatter(zeros, dim=-1, index=idx_d, src=wk_d)
    for t_ in (idx_d, one_d, wk_d):
        ttnn.deallocate(t_)
    hifi = _glue_ckc()
    xg = ttnn.matmul(g_t, hidden_states, transpose_a=True, compute_kernel_config=hifi, dtype=ttnn.bfloat16)  # [1,1,A*M,H]
    ttnn.deallocate(g_t)
    xg = ttnn.reshape(xg, (1, A, M, H))
    ckc = expert_compute_kernel_config()
    gu, dp = weights.gate_up_proj, weights.down_proj
    ot = ttnn.Tile([32, 32])
    ug = ttnn.sparse_matmul(xg, gu, sparsity=sp, indices=ex, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tile=ot,
                            program_config=_build_sparse_matmul_config(M, gu.shape[-1], in0_block_w=32), compute_kernel_config=ckc,
                            is_input_a_sparse=True, dtype=ttnn.bfloat16)
    ttnn.deallocate(xg)
    act = apply_swiglu(ttnn.reshape(ug, (1, A, M, gu.shape[-1])))
    ttnn.deallocate(ug)
    dn = ttnn.sparse_matmul(act, dp, sparsity=sp, indices=ex, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tile=ot,
                            program_config=_build_sparse_matmul_config(M, dp.shape[-1], in0_block_w=16), compute_kernel_config=ckc,
                            is_input_a_sparse=True, dtype=ttnn.bfloat16)
    ttnn.deallocate(act)
    ttnn.deallocate(ex)
    out = ttnn.matmul(w_c, ttnn.reshape(dn, (1, 1, A * M, H)), compute_kernel_config=hifi, dtype=ttnn.bfloat16)
    ttnn.deallocate(w_c)
    ttnn.deallocate(dn)
    return out  # [1, 1, S, H]; rows past valid_len are zero
