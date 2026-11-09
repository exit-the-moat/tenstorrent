# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 pf4: prefill FIR conv (depthwise causal conv1d, K taps) with cheaper data movement.

Same math ops in the same order as the stock _causal_conv1d_fir (multiply, addcmul per tap, silu), so the
output is bit-identical; only the shifted tap inputs are built differently:
- stock: concat(pad, x) in TILE (untilize + concat + tilize), then per tap k >= 1 a non-tile-aligned slice
  of the TILE tensor (untilize + slice + tilize each) and an aligned slice for k = 0;
- QWEN36_FIR_RM=1: untilize x once, concat the K-1 carry rows in ROW_MAJOR, slice + tilize taps 0..K-2,
  and tap K-1 IS x (x_padded[K-1 : K-1+T] = x), no copy at all.
"""
import os

import torch

import ttnn

from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir as _stock


def fir_rm_enabled():
    return os.environ.get("QWEN36_FIR_RM", "0") == "1"


def causal_conv1d_fir(x, weight, bias, kernel_size, device, memory_config=None, conv_state=None,
                      weight_taps=None, bias_dev=None, valid_len=None):
    B, T, D = x.shape[0], x.shape[1], x.shape[2]
    if not (fir_rm_enabled() or fir_k_enabled()) or T == 1 or weight_taps is None or bias is not None or bias_dev is not None:
        return _stock(x, weight, bias, kernel_size, device, memory_config=memory_config, conv_state=conv_state,
                      weight_taps=weight_taps, bias_dev=bias_dev, valid_len=valid_len)
    if fir_k_enabled() and B == 1 and T % 32 == 0 and kernel_size == 4:
        return causal_conv1d_fir_k(x, kernel_size, device, conv_state, weight_taps, valid_len)
    mc = memory_config
    _dram = ttnn.DRAM_MEMORY_CONFIG
    Km1 = kernel_size - 1
    total_len = Km1 + T
    xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    if conv_state is not None:
        pad = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    else:
        pad = ttnn.zeros([B, Km1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=_dram)
    xp = ttnn.concat([pad, xr], dim=1, memory_config=_dram)  # ROW_MAJOR [B, K-1+T, D]
    ttnn.deallocate(pad)
    ttnn.deallocate(xr)

    if valid_len is None:
        new_state = ttnn.slice(xp, (0, total_len - Km1, 0), (B, total_len, D))
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
        new_state = ttnn.to_memory_config(new_state, _dram)
    else:
        # one-hot row select as in the stock path (fixed program per bucket), on the TILE x_padded
        sel = torch.zeros(B, Km1, total_len, dtype=torch.float32)
        if isinstance(valid_len, (list, tuple)):
            for bi in range(B):
                for j in range(Km1):
                    sel[bi, j, int(valid_len[bi]) + j] = 1.0
        else:
            for j in range(Km1):
                sel[:, j, valid_len + j] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        xpt = ttnn.to_layout(xp, ttnn.TILE_LAYOUT, memory_config=_dram)
        new_state = ttnn.matmul(sel_tt, xpt, memory_config=_dram)
        ttnn.deallocate(sel_tt)
        ttnn.deallocate(xpt)

    out = None
    for k in range(kernel_size):
        if k == Km1:
            xs = x
        else:
            xs = ttnn.to_layout(ttnn.slice(xp, (0, k, 0), (B, k + T, D)), ttnn.TILE_LAYOUT, memory_config=mc)
        if out is None:
            out = ttnn.multiply(xs, weight_taps[k], memory_config=mc)
        else:
            out = ttnn.addcmul(out, xs, weight_taps[k], memory_config=mc)
        if k != Km1:
            ttnn.deallocate(xs)
    ttnn.deallocate(xp)
    return ttnn.silu(out, memory_config=_dram), new_state


# ---- QWEN36_FIR_K=1: the whole FIR (shifted tilize + 4-tap MAC + silu) as one program (kernels/firk_*.cpp) ----

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"))
_ONE = 0x3F800000


def fir_k_enabled():
    return os.environ.get("QWEN36_FIR_K", "0") == "1"


class FirK:
    """Per device mesh + tap set: row-replicated taps [1, 1, 4 * 32, D] (tap k = tile row k), built at the first call
    (the eager warm-up, before any trace is captured) and kept."""

    def __init__(self, mesh, taps, K):
        assert K == 4, K
        self.mesh = mesh
        D = taps[0].shape[-1]
        self.D, self.NT = D, D // 32
        self.W = int(os.environ.get("QWEN36_FIR_K_W", "8"))
        assert self.NT % self.W == 0, (self.NT, self.W)
        # TP: each device holds its own channel slice of the taps -> build per device, shard on dim 0
        nd = mesh.get_num_devices()
        per_dev = []
        for d in range(nd):
            rows = []
            for t in taps:
                w = ttnn.to_torch(ttnn.get_device_tensors(t)[d]).reshape(-1)[:D].to(torch.bfloat16)
                rows.append(w.reshape(1, D).expand(32, D))
            per_dev.append(torch.cat(rows, 0).reshape(1, 1, 4 * 32, D))
        self.taps = ttnn.from_torch(
            torch.cat(per_dev, 0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def __call__(self, xr, carry_rm, T, split=0):
        """xr: ROW_MAJOR [1, T, D] (x untilized), carry_rm: ROW_MAJOR [1, 3, D]. -> silu(conv) TILE [1, T, D] DRAM,
        or with split = kd (QWEN36_GDN_FUSE2) the tuple (q [1, T, kd], k [1, T, kd], v [1, T, D - 2 kd])."""
        W, NT, D = self.W, self.NT, self.D
        NR = T // 32
        NU = NR * (NT // W)
        KT = split // 32
        assert split % (32 * W) == 0, (split, W)

        def alloc(w):
            return ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                                  ttnn.DRAM_MEMORY_CONFIG)

        outs = [alloc(split), alloc(split), alloc(D - 2 * split)] if KT else [alloc(D)]
        out = outs[0]
        ko, vo = (outs[1], outs[2]) if KT else (out, out)
        nc = len(self.cores)
        per, rem = divmod(NU, nc)
        rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        u = 0
        for i, c in enumerate(self.cores):
            n = per + (1 if i < rem else 0)
            rr[c.x][c.y] = [xr.buffer_address(), carry_rm.buffer_address(), self.taps.buffer_address(), u, u + n, NR]
            rc[c.x][c.y] = [u, u + n, NR]
            rw[c.x][c.y] = [out.buffer_address(), u, u + n, NR, ko.buffer_address(), vo.buffer_address()]
            u += n
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731

        def kd(src, ct, rt, cfg):
            return ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}",
                                         source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                         core_ranges=self.crs, compile_time_args=ct, runtime_args=rt, config=cfg)

        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False,
                                          math_approx_mode=False)
        cbs = []

        def cb(i, nbytes, page=2048):
            cbs.append(ttnn.CBDescriptor(total_size=nbytes, core_ranges=self.crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)]))

        for k in range(4):
            cb(k, 2 * W * 2048)      # row-major shifted blocks (W tiles each), double-buffered
            cb(4 + k, W * 2048)      # tilized taps inputs
        cb(8, 4 * W * 2048)          # tap weights of the current column block
        cb(9, 35 * W * 64, 35 * W * 64)  # staging: 35 rows x W*64 bytes
        cb(10, 2048)                 # one zero tile
        cb(16, 2 * W * 2048)         # output
        kernels = [
            kd("firk_reader.cpp", [W, NT, D * 2] + acc(xr, carry_rm, self.taps), rr, ttnn.ReaderConfigDescriptor()),
            kd("firk_writer.cpp", [W, NT, KT] + acc(out, ko, vo), rw, ttnn.WriterConfigDescriptor()),
            kd("firk_compute.cpp", [W, _ONE, int(os.environ.get("QWEN36_FIR_K_DBG", "0"))], rc, cc),
        ]
        ttnn.generic_op([xr, carry_rm, self.taps] + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return tuple(outs) if KT else out


_FIRK = {}


def _firk(device, taps, K):
    key = (id(device), id(taps[0]))
    if key not in _FIRK:
        _FIRK[key] = FirK(device, taps, K)
    return _FIRK[key]


def causal_conv1d_fir_k(x, kernel_size, device, conv_state, weight_taps, valid_len, D=None, split=0):
    """QWEN36_FIR_K: bit-identical to the stock FIR (the dev unit test checks it). Masked buckets (valid_len set)
    still build the TILE x_padded for the one-hot state select (as the stock path)."""
    B, T = x.shape[0], x.shape[1]
    _dram = ttnn.DRAM_MEMORY_CONFIG
    Km1 = kernel_size - 1
    fk = _firk(device, weight_taps, kernel_size)
    if D is not None and D != x.shape[2]:
        # QWEN36_GDN_FUSE2: x = the fused qkvzab projection output; qkv = its first D columns, sliced by the untilize
        xr = ttnn.untilize_with_unpadding(x, output_tensor_end=(B - 1, T - 1, D - 1), memory_config=_dram)
    else:
        D = x.shape[2]
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    if conv_state is not None:
        pad = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    else:
        pad = ttnn.zeros([B, Km1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=_dram)
    out = fk(xr, pad, T, split=split)
    if valid_len is None:
        new_state = ttnn.slice(xr, (0, T - Km1, 0), (B, T, D))
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
        new_state = ttnn.to_memory_config(new_state, _dram)
    else:
        total_len = Km1 + T
        sel = torch.zeros(B, Km1, total_len, dtype=torch.float32)
        if isinstance(valid_len, (list, tuple)):
            for bi in range(B):
                for j in range(Km1):
                    sel[bi, j, int(valid_len[bi]) + j] = 1.0
        else:
            for j in range(Km1):
                sel[:, j, valid_len + j] = 1.0
        from models.demos.blackhole.qwen36.tt import mtrace

        _ctx = mtrace.CTX  # traced masked bucket: the one-hot select is a persistent buffer filled per request
        sel_tt = _ctx["fir_sel"] if _ctx is not None else ttnn.from_torch(
            sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        xp = ttnn.concat([pad, xr], dim=1, memory_config=_dram)
        xpt = ttnn.to_layout(xp, ttnn.TILE_LAYOUT, memory_config=_dram)
        ttnn.deallocate(xp)
        new_state = ttnn.matmul(sel_tt, xpt, memory_config=_dram)
        if _ctx is None:
            ttnn.deallocate(sel_tt)
        ttnn.deallocate(xpt)
    ttnn.deallocate(pad)
    ttnn.deallocate(xr)
    return out, new_state


# ---- QWEN36_FIR_T=1: the FIR straight from the TILE qkvzab (full chunks; kernels/firt_*.cpp + firk_writer.cpp) ----


def fir_t_enabled():
    return os.environ.get("QWEN36_FIR_T", "0") == "1"


def _fir_t_call(fk, qkvzab, carry_rm, T, split):
    """qkvzab: TILE [1, T, >= D] (qkv = its first D columns); carry_rm ROW_MAJOR [1, 3, D].
    -> ((q, k, v) TILE DRAM, new carry state ROW_MAJOR [1, 3, D] = the last 3 rows of qkv)."""
    W, NT, D = fk.W, fk.NT, fk.D
    NR = T // 32
    NU = NR * (NT // W)
    KT = split // 32
    NQ = -(-qkvzab.shape[-1] // 32)
    assert split % (32 * W) == 0 and T % 32 == 0, (split, W, T)
    mesh = fk.mesh

    def alloc(w):
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                                              ttnn.DRAM_MEMORY_CONFIG)

    outs = [alloc(split), alloc(split), alloc(D - 2 * split)]
    st = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 3, D]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh,
                                        ttnn.DRAM_MEMORY_CONFIG)
    nc = len(fk.cores)
    per, rem = divmod(NU, nc)
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u = 0
    for i, c in enumerate(fk.cores):
        n = per + (1 if i < rem else 0)
        rr[c.x][c.y] = [qkvzab.buffer_address(), carry_rm.buffer_address(), fk.taps.buffer_address(), u, u + n, NR,
                        st.buffer_address()]
        rc[c.x][c.y] = [u, u + n, NR]
        rw[c.x][c.y] = [outs[0].buffer_address(), u, u + n, NR, outs[1].buffer_address(), outs[2].buffer_address()]
        u += n
    acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731

    def kd(src, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                     core_ranges=fk.crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False)
    cbs = []

    def cb(i, nbytes, page=2048):
        cbs.append(ttnn.CBDescriptor(total_size=nbytes, core_ranges=fk.crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)]))

    for k in range(4):
        cb(k, W * 2048)          # shifted row-major blocks
        cb(4 + k, W * 2048)      # their tilized tiles
    cb(8, 4 * W * 2048)          # taps
    cb(9, 3 * W * 64, 3 * W * 64)  # the previous block's last 3 rows
    cb(10, 2 * W * 2048)         # x tiles in
    cb(11, 2 * W * 2048)         # x untilized (row-major)
    cb(16, 2 * W * 2048)         # output
    kernels = [
        kd("firt_reader.cpp", [W, NT, D * 2, NQ] + acc(qkvzab, carry_rm, fk.taps, st), rr, ttnn.ReaderConfigDescriptor()),
        kd("firk_writer.cpp", [W, NT, KT] + acc(*outs), rw, ttnn.WriterConfigDescriptor()),
        kd("firt_compute.cpp", [W, _ONE, int(os.environ.get("QWEN36_FIR_T_DBG", "0"))], rc, cc),
    ]
    ttnn.generic_op([qkvzab, carry_rm, fk.taps, st] + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return tuple(outs), st


def fir_tm_enabled():
    return os.environ.get("QWEN36_FIR_TM", "0") == "1"


def causal_conv1d_fir_t(qkvzab, kernel_size, device, conv_state, weight_taps, D, split, valid_len=None):
    """Full chunk (no valid_len): bit-identical to causal_conv1d_fir_k(slice(qkvzab, D), ...) (pf4 unit test).
    pf5 QWEN36_FIR_TM (masked buckets, valid_len set): the same conv (it does not depend on valid_len); the new state
    = rows valid_len .. valid_len + K - 2 of [carry; x] as in causal_conv1d_fir_k, by the same one-hot select split
    in two matmuls (sel[:, :K-1] @ carry + sel[:, K-1:] @ qkvzab, then the qkv columns): one 1.0 * x term per output
    element in both, so no concat / tilize of the padded x."""
    _dram = ttnn.DRAM_MEMORY_CONFIG
    fk = _firk(device, weight_taps, kernel_size)
    assert fk.D == D, (fk.D, D)
    T = qkvzab.shape[1]
    Km1 = kernel_size - 1
    if conv_state is not None:
        pad = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    else:
        pad = ttnn.zeros([1, Km1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=_dram)
    (q, k, v), st = _fir_t_call(fk, qkvzab, pad, T, split)
    ttnn.deallocate(pad)
    if valid_len is None:
        new_state = ttnn.to_layout(st, ttnn.TILE_LAYOUT)
        ttnn.deallocate(st)
        new_state = ttnn.to_memory_config(new_state, _dram)
        return (q, k, v), new_state
    ttnn.deallocate(st)
    from models.demos.blackhole.qwen36.tt import mtrace

    _ctx = mtrace.CTX  # traced masked bucket: the one-hot select is a persistent buffer filled per request
    if _ctx is not None:
        sel_tt = _ctx["fir_sel"]
    else:
        sel = torch.zeros(1, Km1, Km1 + T, dtype=torch.float32)
        for j in range(Km1):
            sel[:, j, valid_len + j] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _split = _ctx is not None and "fir_sel_x" in _ctx and os.environ.get("QWEN36_FIR_TM_SPLIT") == "1"
    if _split:  # pf5: the two parts filled per request (model._mtrace_*), no unaligned device slice
        sel_p, sel_x = _ctx["fir_sel_p"], _ctx["fir_sel_x"]
    else:
        sel_p = ttnn.slice(sel_tt, (0, 0, 0), (1, Km1, Km1), memory_config=_dram)
        sel_x = ttnn.slice(sel_tt, (0, 0, Km1), (1, Km1, Km1 + T), memory_config=_dram)
    if _ctx is None:
        ttnn.deallocate(sel_tt)
    if conv_state is not None:
        cs_t = conv_state
    else:
        cs_t = ttnn.zeros([1, Km1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram)
    ns_x = ttnn.matmul(sel_x, qkvzab, memory_config=_dram)
    ns_p = ttnn.matmul(sel_p, cs_t, memory_config=_dram)
    if not _split:
        ttnn.deallocate(sel_x)
        ttnn.deallocate(sel_p)
    if conv_state is None:
        ttnn.deallocate(cs_t)
    ns_xq = ttnn.slice(ns_x, (0, 0, 0), (1, Km1, D), memory_config=_dram)
    ttnn.deallocate(ns_x)
    new_state = ttnn.add(ns_xq, ns_p, memory_config=_dram)
    ttnn.deallocate(ns_xq)
    ttnn.deallocate(ns_p)
    return (q, k, v), new_state
