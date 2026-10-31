# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused one-token gated-attention pre-SDPA step (port5 Track B, opt-in: DRAMA3_ATTN_FUSED_PRE=1).

The decode attention (backbone and tower, Qwen36GatedAttention, 16 q / 2 kv heads of 256, partial rotary 64) runs
~20 ops between the q|gate, k, v projections and paged_update_cache: chunk / reshapes, the q / k rms norms, four
transposes, the rotary (x * C + (x @ P) * S per tensor), the sharding of k / v, q to DRAM. Here that is ONE program
(kernels/attn_pre_*.cpp): the head rows are packed straight into the [heads, 256] tile layout SDPA decode and
paged_update_cache take (row h = head h), normed per row and rotated on whole tiles (the half rotation of the 64
rotary dims swaps tiles 0 and 1), on 3 cores (q, k, v) + 4 cores copying the gate tiles out of the q|gate row.
The cache update, SDPA, head concat, sigmoid gate and o_proj stay ttnn ops.
"""

import os
import struct

import torch

import ttnn

KDIR = "models/demos/blackhole/drama3/kernels"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class AttnPreFused:
    def __init__(self, attn, mesh):
        cfg = attn.config
        assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (16, 2, 256), (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)
        self.attn, self.mesh = attn, mesh
        up = lambda t: ttnn.from_torch(  # noqa: E731
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w = attn.weights
        self.wq = up(ttnn.to_torch(w.q_norm).float().flatten()[:256].reshape(1, 1, 256))
        self.wk = up(ttnn.to_torch(w.k_norm).float().flatten()[:256].reshape(1, 1, 256))
        self.ones = up(torch.ones(32, 32))
        self.ct = [_bits(1.0 / 256), _bits(cfg.norm_eps)]
        C = ttnn.CoreCoord
        self.cq, self.ck, self.cv = C(1, 0), C(2, 0), C(3, 0)
        self.cgp = C(4, 0)  # the packed gate (DRAMA3_ATTN_FUSED_POST) or the first of 4 gate-copy cores
        self.cg = [C(4 + i, 0) for i in range(4)]
        self.post = os.environ.get("DRAMA3_ATTN_FUSED_POST", "0") == "1"
        # DRAMA3_ATTN_QKV_MERGED=1: [q|gate | k | v] as one weight (one one-token matmul instead of three)
        self.w_qkv = None
        if os.environ.get("DRAMA3_ATTN_QKV_MERGED", "0") == "1":
            parts = [ttnn.to_torch(t).float().reshape(-1, t.shape[-1]) for t in (w.q_proj, w.k_proj, w.v_proj)]
            self.w_qkv = ttnn.from_torch(torch.cat(parts, -1), dtype=w.q_proj.dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        one = lambda c: ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])  # noqa: E731
        self.rq, self.rk, self.rv = one(self.cq), one(self.ck), one(self.cv)
        self.rg = ttnn.CoreRangeSet([ttnn.CoreRange(self.cg[0], self.cg[-1])])
        self.rqk = ttnn.CoreRangeSet([ttnn.CoreRange(self.cq, self.ck)])
        self.rall = ttnn.CoreRangeSet([ttnn.CoreRange(self.cq, self.cg[-1])])
        shard = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(C(0, 0), C(0, 0))}), [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
        self.kv_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

    def __call__(self, qg, k, v, cos, sin, kbase=0, vbase=0):
        """qg [1, 1, 16 * 512] (q | gate per head), k, v [1, 1, 2 * 256] (L1) -> (q [1, 1, 16, 256] DRAM, k / v
        [1, 1, 2, 256] height-sharded L1, gate [1, 1, 4096] L1)."""
        m = self.mesh
        q_dec = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 16, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG)
        k_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, self.kv_mc)
        v_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, self.kv_mc)
        if self.post:  # packed [16 heads, 256] like q
            gate = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 16, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
        else:
            gate = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)

        def acc(*ts):
            out = []
            for t in ts:
                out.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            return out

        def kd(src, crs, ct, rt, cfg):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/{src}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )

        def rt_one(c, args):
            r = ttnn.RuntimeArgs()
            r[c.x][c.y] = args
            return r

        kernels = []
        # readers: q (16 heads, 16 source tiles per head), k / v (2 heads, 8), gate copies
        for role, c, crs, src, nh, stride, w, dst, base in (
            (0, self.cq, self.rq, qg, 16, 16, self.wq, q_dec, 0),
            (1, self.ck, self.rk, k, 2, 8, self.wk, k_sh, kbase),
            (2, self.cv, self.rv, v, 2, 8, self.wk, v_sh, vbase),
        ):
            kernels.append(
                kd(
                    "attn_pre_reader.cpp",
                    crs,
                    [role, nh, stride] + acc(src, w, cos, sin, self.ones, dst),
                    rt_one(c, [src.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                               self.ones.buffer_address(), dst.buffer_address(), 0, 0, base]),
                    ttnn.ReaderConfigDescriptor(),
                )
            )
        if self.post:
            rgp = ttnn.CoreRangeSet([ttnn.CoreRange(self.cgp, self.cgp)])
            kernels.append(kd("attn_pre_reader.cpp", rgp, [4, 16, 16] + acc(qg, qg, qg, qg, qg, gate),
                              rt_one(self.cgp, [qg.buffer_address(), 0, 0, 0, 0, gate.buffer_address(), 0, 0, 0]), ttnn.ReaderConfigDescriptor()))
            kernels.append(kd("attn_pre_writer.cpp", rgp, acc(gate), rt_one(self.cgp, [gate.buffer_address()]), ttnn.WriterConfigDescriptor()))
        else:
            rg = ttnn.RuntimeArgs()
            for i, c in enumerate(self.cg):
                rg[c.x][c.y] = [qg.buffer_address(), 0, 0, 0, 0, gate.buffer_address(), 32 * i, 32, 0]
            kernels.append(kd("attn_pre_reader.cpp", self.rg, [3, 16, 16] + acc(qg, qg, qg, qg, qg, gate), rg, ttnn.ReaderConfigDescriptor()))
        # compute: q and k cores
        kernels.append(
            kd("attn_pre_compute.cpp", self.rqk, self.ct, [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False))
        )
        # writers: q to DRAM, k / v to the sharded cache inputs
        kernels.append(kd("attn_pre_writer.cpp", self.rq, acc(q_dec), rt_one(self.cq, [q_dec.buffer_address()]), ttnn.WriterConfigDescriptor()))
        rkv = ttnn.RuntimeArgs()
        rkv[self.ck.x][self.ck.y] = [k_sh.buffer_address()]
        rkv[self.cv.x][self.cv.y] = [v_sh.buffer_address()]
        kernels.append(kd("attn_pre_writer.cpp", ttnn.CoreRangeSet([ttnn.CoreRange(self.ck, self.cv)]), acc(k_sh), rkv, ttnn.WriterConfigDescriptor()))
        sizes = {0: 32, 1: 8, 2: 4, 3: 1, 16: 8, 24: 8, 25: 1, 26: 8, 27: 8}
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.rall,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in sizes.items()
        ]
        io = [qg, k, v, self.wq, self.wk, self.ones, cos, sin, q_dec, k_sh, v_sh, gate]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return q_dec, k_sh, v_sh, gate


def _batch_pre(self, qkv, cos, sin, B, kbase, vbase):
    """port6 (DRAMA3_ATTN_FUSED_BATCH=1, with the merged qkv and the fused post): AttnPreFused for B decode rows,
    qkv [1, B, N] (L1), cos / sin [B, 1, 256] -> (q [1, B, 16, 256] DRAM, k / v [1, B, 2, 256] height-sharded on B
    cores, the packed gate [1, B, 16, 256] L1). Four cores per row (q, k, v, gate: kernels/attn_pre_b_*.cpp)."""
    m, C = self.mesh, ttnn.CoreCoord
    g = m.compute_with_storage_grid_size()
    core = lambda i: C(i % g.x, i // g.x)  # noqa: E731
    # port10: B cores row-major over the grid (one grid row held only g.x <= 11..13 rows)
    kv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(C(0, 0), C(B - 1, 0))}) if B <= g.x else ttnn.num_cores_to_corerangeset(B, g, row_wise=True)
    shard = ttnn.ShardSpec(kv_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
    kv_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    q_dec = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 16, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG)
    k_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 2, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
    v_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 2, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
    gate = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 16, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
    acc = lambda *ts: [x for t in ts for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
    sets = [ttnn.CoreRangeSet([ttnn.CoreRange(core(4 * b + r), core(4 * b + r)) for b in range(B)]) for r in range(4)]
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(core(i), core(i)) for i in range(4 * B)])
    qk = ttnn.CoreRangeSet([ttnn.CoreRange(core(4 * b + r), core(4 * b + r)) for b in range(B) for r in range(2)])
    kv = ttnn.CoreRangeSet([ttnn.CoreRange(core(4 * b + r), core(4 * b + r)) for b in range(B) for r in (1, 2)])

    def kd(src, crs, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                     core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    kernels = []
    roles = ((0, qkv, 16, 16, self.wq, q_dec, 0), (1, qkv, 2, 8, self.wk, k_sh, kbase), (2, qkv, 2, 8, self.wk, v_sh, vbase),
             (4, qkv, 16, 16, qkv, gate, 0))
    for i, (role, src, nh, stride, w, dst, base) in enumerate(roles):
        rt = ttnn.RuntimeArgs()
        for b in range(B):
            c = core(4 * b + i)
            if role == 4:
                rt[c.x][c.y] = [src.buffer_address(), 0, 0, 0, 0, dst.buffer_address(), 0, 0, 0, b]
            else:
                rt[c.x][c.y] = [src.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                                self.ones.buffer_address(), dst.buffer_address(), 0, 0, base, b]
        ct = [role, nh, stride] + (acc(src, src, src, src, src, dst) if role == 4 else acc(src, w, cos, sin, self.ones, dst))
        kernels.append(kd("attn_pre_b_reader.cpp", sets[i], ct, rt, ttnn.ReaderConfigDescriptor()))
    kernels.append(kd("attn_pre_compute.cpp", qk, self.ct, [],
                      ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)))
    for crs, rows, ct in ((sets[0], [(0, q_dec)], acc(q_dec)), (kv, [(1, k_sh), (2, v_sh)], acc(k_sh)), (sets[3], [(3, gate)], acc(gate))):
        rt = ttnn.RuntimeArgs()
        for b in range(B):
            for r, t in rows:
                c = core(4 * b + r)
                rt[c.x][c.y] = [t.buffer_address(), b]
        kernels.append(kd("attn_pre_b_writer.cpp", crs, ct, rt, ttnn.WriterConfigDescriptor()))
    sizes = {0: 32, 1: 8, 2: 4, 3: 1, 16: 8, 24: 8, 25: 1, 26: 8, 27: 8}
    cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=allc,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
           for i, n in sizes.items()]
    io = [qkv, self.wq, self.wk, self.ones, cos, sin, q_dec, k_sh, v_sh, gate]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return q_dec, k_sh, v_sh, gate


def _batch_post(self, attn, gpk, B):
    """port6: AttnPostFused for B decode rows: [1, B, 16, 256] SDPA out x sigmoid(packed gate) -> [1, B, 4096] (L1),
    8 cores per row (kernels/attn_post_b_*.cpp; the compute kernel is the one-token one)."""
    m = self.mesh
    g = m.compute_with_storage_grid_size()
    if 8 * B > g.x * g.y:
        return _batch_post2(self, attn, gpk, B)
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(8 * B)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        b, j = i // 8, i % 8
        rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address(), b]
        rw[c.x][c.y] = [j, out.buffer_address(), b, B]
    kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct, runtime_args=rt, config=cfg,
    )
    kernels = [
        kd("attn_post_b_reader.cpp", acc(attn) + acc(gpk), rr, ttnn.ReaderConfigDescriptor()),
        kd("attn_post_b_writer.cpp", [16] + acc(out), rw, ttnn.WriterConfigDescriptor()),
        kd("attn_post_compute.cpp", [], [],
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
    ]
    cbs = [ttnn.CBDescriptor(total_size=2048, core_ranges=crs,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
           for i in (0, 1, 3, 16)]
    ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def _batch_post2(self, attn, gpk, B):
    """port10: _batch_post when 8 B cores do not fit the grid (B <= 32): 8 G cores, G = cores // 8 row groups, core
    (j, g0) runs rows g0, g0 + G, .. (kernels/attn_post_b2_*.cpp)."""
    m = self.mesh
    g = m.compute_with_storage_grid_size()
    assert B <= 32
    G = (g.x * g.y) // 8
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(8 * G)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        g0, j = i // 8, i % 8
        rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address(), g0, G, B]
        rw[c.x][c.y] = [j, out.buffer_address(), g0, G, B]
        rc[c.x][c.y] = [len(range(g0, B, G))]
    kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct, runtime_args=rt, config=cfg,
    )
    kernels = [
        kd("attn_post_b2_reader.cpp", acc(attn) + acc(gpk), rr, ttnn.ReaderConfigDescriptor()),
        kd("attn_post_b2_writer.cpp", [16] + acc(out), rw, ttnn.WriterConfigDescriptor()),
        kd("attn_post_b2_compute.cpp", [], rc,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
    ]
    cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=crs,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
           for i, n in ((0, 2), (1, 2), (3, 1), (16, 2))]
    ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


class AttnPostFused:
    """DRAMA3_ATTN_FUSED_POST=1 (with DRAMA3_ATTN_FUSED_PRE): SDPA output [1, 1, 16, 256] (row h = head h) times
    sigmoid(gate) (packed the same way by AttnPreFused) and unpacked into the [1, 1, 4096] o_proj input row
    (kernels/attn_post_*.cpp, 8 cores, one per column tile) instead of transpose + concatenate_heads + sigmoid + mul."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.cores = [ttnn.CoreCoord(j, 1) for j in range(8)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])

    def __call__(self, attn, gpk):
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for j, c in enumerate(self.cores):
            rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address()]
            rw[c.x][c.y] = [j, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("attn_post_reader.cpp", acc(attn) + acc(gpk), rr, ttnn.ReaderConfigDescriptor()),
            kd("attn_post_writer.cpp", [16] + acc(out), rw, ttnn.WriterConfigDescriptor()),
            kd("attn_post_compute.cpp", [], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in {0: 1, 1: 1, 2: 16, 3: 1, 16: 1}.items()
        ]
        ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


def install(modules, mesh):
    """Route the one-token paged decode of every given Qwen36GatedAttention through AttnPreFused."""
    if os.environ.get("DRAMA3_ATTN_FUSED_PRE", "0") != "1":
        return 0
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import _get_paged_sdpa_decode_program_config

    post = AttnPostFused(mesh) if os.environ.get("DRAMA3_ATTN_FUSED_POST", "0") == "1" else None
    n = 0
    for at in modules:
        pre = AttnPreFused(at, mesh)
        orig = getattr(at, "_port5_orig_forward", at.forward)  # a second install wraps the original, not the first wrapper
        at._port5_orig_forward = orig

        def forward(x, cos, sin, position_tensor=None, page_table=None, chunk_page_table=None, chunk_start_idx=None,
                    chunk_start_idx_tensor=None, _a=at, _o=orig, _p=pre, _post=post):
            if (getattr(_a, "rows_are_batch", False) and batch_on() and _post is not None and _p.w_qkv is not None
                    and _a.use_paged_attention and chunk_page_table is None and cos.shape[-1] == 256):
                return _batch_forward(_a, _p, _post, x, cos, sin, position_tensor, page_table)
            if (not _a.use_paged_attention or x.shape[1] != 1 or getattr(_a, "rows_are_batch", False)
                    or chunk_page_table is not None or cos.shape[-1] != 256):
                return _o(x, cos, sin, position_tensor=position_tensor, page_table=page_table, chunk_page_table=chunk_page_table,
                          chunk_start_idx=chunk_start_idx, chunk_start_idx_tensor=chunk_start_idx_tensor)
            w, L1, ckc = _a.weights, ttnn.L1_MEMORY_CONFIG, _a.compute_kernel_config_decode
            if _p.w_qkv is not None:  # one [q|gate | k | v] row: k at page 256, v at 272
                qkv = ttnn.linear(x, _p.w_qkv, compute_kernel_config=ckc, memory_config=L1)
                q_dec, k_sh, v_sh, gate = _p(qkv, qkv, qkv, cos, sin, 256, 272)
                ttnn.deallocate(qkv)
            else:
                qg = ttnn.linear(x, w.q_proj, compute_kernel_config=ckc, memory_config=L1)
                k = ttnn.linear(x, w.k_proj, compute_kernel_config=ckc, memory_config=L1)
                v = ttnn.linear(x, w.v_proj, compute_kernel_config=ckc, memory_config=L1)
                q_dec, k_sh, v_sh, gate = _p(qg, k, v, cos, sin)
                for t_ in (qg, k, v):
                    ttnn.deallocate(t_)
            K, V = _a.paged_kv_cache_key, _a.paged_kv_cache_value
            ttnn.experimental.paged_update_cache(K, k_sh, update_idxs_tensor=position_tensor, page_table=page_table)
            ttnn.experimental.paged_update_cache(V, v_sh, update_idxs_tensor=position_tensor, page_table=page_table)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_dec, K, V, cur_pos_tensor=position_tensor, page_table_tensor=page_table, is_causal=True,
                scale=_a.config.head_dim**-0.5,
                program_config=_get_paged_sdpa_decode_program_config(_a.device, K.shape[0] * K.shape[2]),
            )
            for t_ in (q_dec, k_sh, v_sh):
                ttnn.deallocate(t_)
            if _post is not None:
                o = _post(attn, gate)
                ttnn.deallocate(attn)
                ttnn.deallocate(gate)
                return ttnn.linear(o, w.o_proj, compute_kernel_config=ckc, memory_config=L1)
            attn = ttnn.transpose(attn, 1, 2)
            attn = ttnn.transformer.concatenate_heads(attn)
            if os.environ.get("QWEN36_ATTN_GATE_FUSED", "0") == "1":
                attn = ttnn.multiply(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            else:
                g = ttnn.sigmoid(gate)
                attn = ttnn.multiply(attn, g)
                ttnn.deallocate(g)
            ttnn.deallocate(gate)
            return ttnn.linear(attn, w.o_proj, compute_kernel_config=ckc, memory_config=L1)

        at.forward = forward
        n += 1
    return n


def batch_on():
    """DRAMA3_ATTN_FUSED_BATCH=1 (port6, opt-in; needs DRAMA3_ATTN_FUSED_POST and DRAMA3_ATTN_QKV_MERGED): batch-mode
    decode attention (B one-token sequences, backbone and tower) through the batch pre / post kernels."""
    return os.environ.get("DRAMA3_ATTN_FUSED_BATCH", "0") == "1"


def _batch_forward(a, pre, post, x, cos, sin, position_tensor, page_table):
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import _get_paged_sdpa_decode_program_config

    w, L1, ckc = a.weights, ttnn.L1_MEMORY_CONFIG, a.compute_kernel_config_decode
    B = x.shape[-2]
    qkv = ttnn.linear(x, pre.w_qkv, compute_kernel_config=ckc, memory_config=L1)  # [1, B, q|gate | k | v]
    q_dec, k_sh, v_sh, gate = _batch_pre(pre, qkv, cos, sin, B, 256, 272)
    ttnn.deallocate(qkv)
    K, V = a.paged_kv_cache_key, a.paged_kv_cache_value
    ttnn.experimental.paged_update_cache(K, k_sh, update_idxs_tensor=position_tensor, page_table=page_table)
    ttnn.experimental.paged_update_cache(V, v_sh, update_idxs_tensor=position_tensor, page_table=page_table)
    att = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_dec, K, V, cur_pos_tensor=position_tensor, page_table_tensor=page_table, is_causal=True,
        scale=a.config.head_dim**-0.5, program_config=_get_paged_sdpa_decode_program_config(a.device, K.shape[0] * K.shape[2]),
    )
    for t in (q_dec, k_sh, v_sh):
        ttnn.deallocate(t)
    o = _batch_post(post, att, gate, B)
    ttnn.deallocate(att)
    ttnn.deallocate(gate)
    y = ttnn.linear(o, w.o_proj, compute_kernel_config=ckc, memory_config=L1)
    ttnn.deallocate(o)
    return y
