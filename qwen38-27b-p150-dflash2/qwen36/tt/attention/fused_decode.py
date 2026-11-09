# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 (Exit the Moat, Episode 2): fused one-token gated-attention pre / post SDPA steps for the 1-card decode
(Qwen36GatedAttention, paged, B = 1; opt-in: QWEN36_ATTN_FUSED=1).

The paged decode branch of gated_attention_forward_ttnn runs ~35 small ops around the cache update and SDPA: chunk /
reshapes of q|gate, the q / k rms norms, four transposes, the partial rotary (slices, rotate_half = slice + neg +
concat, 4 multiplies, 2 adds, 2 concats per tensor), the sharding of k / v, q to DRAM; after SDPA a transpose,
concatenate_heads, sigmoid and multiply. Here: ONE pre program (kernels/attnp_*.cpp: the head rows packed straight
into the [heads, 256] tile layout SDPA decode and paged_update_cache take, normed per row, the 64 rotary dims rotated
on whole tiles; 4 cores: q, k, v, packed gate) and ONE post program (kernels/attnpo_*.cpp: SDPA out x sigmoid(gate),
unpacked into the o_proj input row; 8 cores). Port of the Episode 0 Drama 3 kernels (16 / 2 heads) to 24 / 4 heads.
Projections, paged_update_cache, paged SDPA decode and o_proj stay ttnn ops (same calls as the chain).
"""

import os
import struct

import torch

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"))


def enabled():
    return os.environ.get("QWEN36_ATTN_FUSED", "0") == "1"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _cb(crs, sizes):
    return [
        ttnn.CBDescriptor(
            total_size=(4096 if f32 else 2048) * n,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=ttnn.float32 if f32 else ttnn.bfloat16, page_size=4096 if f32 else 2048
                )
            ],
        )
        for i, (n, f32) in sizes.items()
    ]


class AttnFused1:
    def __init__(self, attn):
        cfg = attn.config
        self.attn, self.mesh = attn, attn.device
        self.NH, self.NKV, self.D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        assert self.D == 256 and self.NH <= 32 and self.NKV <= 32, (self.NH, self.NKV, self.D)
        mesh = self.mesh

        def up(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w = attn.weights
        self.wq = up(ttnn.to_torch(w.q_norm).float().flatten()[:256].reshape(1, 1, 256))
        self.wk = up(ttnn.to_torch(w.k_norm).float().flatten()[:256].reshape(1, 1, 256))
        self.ones = up(torch.ones(32, 32))
        self.ct = [_bits(1.0 / 256), _bits(cfg.norm_eps)]
        C = ttnn.CoreCoord
        self.cq, self.ck, self.cv, self.cg = C(1, 0), C(2, 0), C(3, 0), C(4, 0)
        one = lambda c: ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])  # noqa: E731
        self.r = {c: one(c) for c in (self.cq, self.ck, self.cv, self.cg)}
        self.rqk = ttnn.CoreRangeSet([ttnn.CoreRange(self.cq, self.ck)])
        self.rkv = ttnn.CoreRangeSet([ttnn.CoreRange(self.ck, self.cv)])
        self.rall = ttnn.CoreRangeSet([ttnn.CoreRange(self.cq, self.cg)])
        shard = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(C(0, 0), C(0, 0))}), [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
        self.kv_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
        self.post_cores = [C(j, 1) for j in range(8)]
        self.rpost = ttnn.CoreRangeSet([ttnn.CoreRange(self.post_cores[0], self.post_cores[-1])])

    @staticmethod
    def _acc(*ts):
        return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    @staticmethod
    def _kd(src, crs, ct, rt, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )

    @staticmethod
    def _rt(c, args):
        r = ttnn.RuntimeArgs()
        r[c.x][c.y] = args
        return r

    def pre(self, qg, k, v, cos, sin):
        """qg [1, 1, NH * 512] (q | gate per head), k, v [1, 1, NKV * 256] -> (q [1, 1, NH, 256] DRAM, k / v
        [1, 1, NKV, 256] height-sharded L1 on core (0, 0), the packed gate [1, 1, NH, 256] L1)."""
        m, NH, NKV = self.mesh, self.NH, self.NKV
        q_dec = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG)
        k_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NKV, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, self.kv_mc)
        v_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NKV, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, self.kv_mc)
        gate = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
        kd, acc, rt = self._kd, self._acc, self._rt
        kernels = []
        for role, c, src, nh, stride, w in (
            (0, self.cq, qg, NH, 16, self.wq),
            (1, self.ck, k, NKV, 8, self.wk),
            (2, self.cv, v, NKV, 8, self.wk),
            (4, self.cg, qg, NH, 16, self.wq),
        ):
            kernels.append(kd(
                "attnp_reader.cpp", self.r[c], [role, nh, stride, 8] + acc(src, w, cos, sin, self.ones),
                rt(c, [src.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                       self.ones.buffer_address(), 0, 0, 0]),
                ttnn.ReaderConfigDescriptor(),
            ))
        kernels.append(kd(
            "attnp_compute.cpp", self.rqk, self.ct, [],
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
        ))
        kernels.append(kd("attnp_writer.cpp", self.r[self.cq], acc(q_dec), rt(self.cq, [q_dec.buffer_address(), 0]),
                          ttnn.WriterConfigDescriptor()))
        rkv = ttnn.RuntimeArgs()
        rkv[self.ck.x][self.ck.y] = [k_sh.buffer_address(), 0]
        rkv[self.cv.x][self.cv.y] = [v_sh.buffer_address(), 0]
        kernels.append(kd("attnp_writer.cpp", self.rkv, acc(k_sh), rkv, ttnn.WriterConfigDescriptor()))
        kernels.append(kd("attnp_writer.cpp", self.r[self.cg], acc(gate), rt(self.cg, [gate.buffer_address(), 0]),
                          ttnn.WriterConfigDescriptor()))
        sizes = {0: (8, 0), 1: (8, 0), 2: (4, 0), 3: (1, 0), 5: (12, 0), 16: (8, 0), 24: (8, 1), 25: (1, 1), 26: (8, 1), 27: (8, 1)}
        io = [qg, k, v, self.wq, self.wk, self.ones, cos, sin, q_dec, k_sh, v_sh, gate]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=_cb(self.rall, sizes)))
        return q_dec, k_sh, v_sh, gate

    def post(self, attn, gpk):
        """attn: SDPA out [1, 1, NH, 256] (row h = head h), gpk: the packed gate -> [1, 1, NH * 256] (L1)."""
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.NH * 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh, ttnn.L1_MEMORY_CONFIG)
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for j, c in enumerate(self.post_cores):
            rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address()]
            rw[c.x][c.y] = [j, out.buffer_address()]
        kd, acc = self._kd, self._acc
        kernels = [
            kd("attnpo_reader.cpp", self.rpost, acc(attn, gpk), rr, ttnn.ReaderConfigDescriptor()),
            kd("attnpo_writer.cpp", self.rpost, [self.NH] + acc(out), rw, ttnn.WriterConfigDescriptor()),
            kd("attnpo_compute.cpp", self.rpost, [], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: (1, 0), 1: (1, 0), 2: (self.NH, 0), 3: (1, 0), 16: (1, 0)}
        ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=_cb(self.rpost, sizes)))
        return out

    def ok(self, x, cos, page_table, key):
        return (
            x.shape[0] == 1 and x.shape[1] == 1 and page_table is not None and key is not None
            and cos.shape[-1] == 64 and x.dtype == ttnn.bfloat16
        )

    def forward(self, x, cos, sin, w, ckc, mc, position_tensor, page_table, key, value):
        """The paged one-token decode (same projections, cache update, SDPA call and o_proj as the chain)."""
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            _get_paged_sdpa_decode_program_config,
        )

        qg = ttnn.linear(x, w.q_proj, compute_kernel_config=ckc, memory_config=mc)
        k = ttnn.linear(x, w.k_proj, compute_kernel_config=ckc, memory_config=mc)
        v = ttnn.linear(x, w.v_proj, compute_kernel_config=ckc, memory_config=mc)
        q_dec, k_sh, v_sh, gpk = self.pre(qg, k, v, cos, sin)
        for t in (qg, k, v):
            ttnn.deallocate(t)
        ttnn.experimental.paged_update_cache(key, k_sh, update_idxs_tensor=position_tensor, page_table=page_table)
        ttnn.experimental.paged_update_cache(value, v_sh, update_idxs_tensor=position_tensor, page_table=page_table)
        ttnn.deallocate(k_sh)
        ttnn.deallocate(v_sh)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_dec, key, value, cur_pos_tensor=position_tensor, page_table_tensor=page_table, is_causal=True,
            scale=self.D**-0.5,
            program_config=_get_paged_sdpa_decode_program_config(self.mesh, key.shape[0] * key.shape[2]),
        )
        ttnn.deallocate(q_dec)
        o_in = self.post(attn, gpk)
        ttnn.deallocate(attn)
        ttnn.deallocate(gpk)
        out = ttnn.linear(o_in, w.o_proj, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(o_in)
        return out


class AttnFusedTP(AttnFused1):
    """TPAttention (the 2-card path): per device NH q / NKV kv heads, fused [q | k | v | gate] decode projection
    (qkv3 [1, 1, B, NH*256 + 2*NKV*256] and gate [1, 1, B, NH*256]), B <= 8 sequences (rows). Pre: 4 cores per
    sequence (q, k, v, gate); outputs q [1, B, NH, 256] DRAM, k / v [1, B, 32, 256] in the paged_update_cache height
    shard, the packed gate [1, B, NH, 256] L1. Post: 8 cores (one per column tile) for all sequences -> [1, B, NH*256]."""

    def __init__(self, at):
        self.at, self.mesh = at, at.mesh
        self.NH, self.NKV, self.D = at.NH, at.NKV, at.HD
        assert self.D == 256 and at.rope_dim == 64 and self.NH <= 32 and self.NKV <= 32, (self.NH, self.NKV, self.D)
        mesh = self.mesh
        nd = mesh.get_num_devices()

        def up(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if nd > 1 else None,
            )

        def row(t):  # a replicated [.., 256] weight -> [1, 1, 256] (device 0's copy)
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().flatten()[:256].reshape(1, 1, 256)

        self.wq = up(row(at.tw["q_norm"]))
        self.wk = up(row(at.tw["k_norm"]))
        self.ones = up(torch.ones(32, 32))
        self.ct = [_bits(1.0 / 256), _bits(1e-6)]
        self.grid = mesh.compute_with_storage_grid_size()

    def _core(self, i):
        return ttnn.CoreCoord(i % self.grid.x, i // self.grid.x)

    def _span(self, i0, n):
        """ep2 gdn: core indices [i0, i0 + n) as at most 3 row rectangles (dispatch pays per core range: single-core
        ranges made the split programs slower than the originals)."""
        gx, rs, i = self.grid.x, [], i0
        while i < i0 + n:
            y, x0 = i // gx, i % gx
            if x0 == 0 and i0 + n - i >= gx:  # whole rows
                rows = (i0 + n - i) // gx
                rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(gx - 1, y + rows - 1)))
                i += rows * gx
            else:
                x1 = min(gx, x0 + (i0 + n - i))
                rs.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1 - 1, y)))
                i += x1 - x0
        return ttnn.CoreRangeSet(rs)

    def ok_tp(self, qkv3, gate, cos):
        return (
            qkv3.dtype == ttnn.bfloat16 and gate.dtype == ttnn.bfloat16 and qkv3.layout == ttnn.TILE_LAYOUT
            and cos.layout == ttnn.TILE_LAYOUT and cos.shape[-1] == 64 and not qkv3.is_sharded() and not gate.is_sharded()
        )

    def pre_tp(self, qkv3, gate, cos, sin, B, kv_mc):
        QS = int(os.environ.get("QWEN36_ATTN_SPLIT", "0"))
        # the largest split <= QS with whole, even head blocks that fits the grid ((2 QS + 2) B cores)
        while QS >= 1 and (self.NH % QS or (self.NH // QS) % 2 or (2 * QS + 2) * B > self.grid.x * self.grid.y):
            QS -= 1
        if QS >= 1 and self.NH <= 16:
            return self._pre_tp_split(qkv3, gate, cos, sin, B, kv_mc, QS)
        QS0 = int(os.environ.get("QWEN36_ATTN_SPLIT", "0"))
        if QS0 >= 1 and self.NH <= 16 and B > 16:
            # t3c: B > 16 does not fit one program: groups of 16 sequences (each its own split program, rows are
            # per-row work: the same output as one program)
            m, NH = self.mesh, self.NH
            outs = (
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG),
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc),
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc),
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG),
            )
            for b0 in range(0, B, 16):
                bg = min(16, B - b0)
                q = QS0
                while q >= 1 and (self.NH % q or (self.NH // q) % 2 or (2 * q + 2) * bg > self.grid.x * self.grid.y):
                    q -= 1
                assert q >= 1, (B, bg)
                self._pre_tp_split(qkv3, gate, cos, sin, B, kv_mc, q, b0=b0, bg=bg, outs=outs)
            return outs
        m, NH, NKV = self.mesh, self.NH, self.NKV
        q_dec = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG)
        k_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
        v_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
        gpk = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
        kd, acc = self._kd, self._acc
        crs = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        sets = [[self._core(4 * b + r) for b in range(B)] for r in range(4)]
        kernels = []
        # role, source, heads, stride, gate offset, norm weight, first source page, destination
        roles = (
            (0, qkv3, NH, 8, 0, self.wq, 0, q_dec),
            (1, qkv3, NKV, 8, 0, self.wk, 8 * NH, k_sh),
            (2, qkv3, NKV, 8, 0, self.wk, 8 * NH + 8 * NKV, v_sh),
            (4, gate, NH, 8, 0, self.wq, 0, gpk),
        )
        for r, (role, src, nh, stride, goff, w, base, dst) in enumerate(roles):
            rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for b, c in enumerate(sets[r]):
                rr[c.x][c.y] = [src.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                                self.ones.buffer_address(), base, b, 2 * b]
                rw[c.x][c.y] = [dst.buffer_address(), 8 * b]
            kernels.append(kd("attnp_reader.cpp", crs(sets[r]), [role, nh, stride, goff] + acc(src, w, cos, sin, self.ones),
                              rr, ttnn.ReaderConfigDescriptor()))
            kernels.append(kd("attnp_writer.cpp", crs(sets[r]), acc(dst), rw, ttnn.WriterConfigDescriptor()))
        kernels.append(kd(
            "attnp_compute.cpp", crs(sets[0] + sets[1]), self.ct, [],
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
        ))
        sizes = {0: (8, 0), 1: (8, 0), 2: (4, 0), 3: (1, 0), 5: (12, 0), 16: (8, 0), 24: (8, 1), 25: (1, 1), 26: (8, 1), 27: (8, 1)}
        io = [qkv3, gate, self.wq, self.wk, self.ones, cos, sin, q_dec, k_sh, v_sh, gpk]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=_cb(crs([c for s in sets for c in s]), sizes)))
        return q_dec, k_sh, v_sh, gpk

    def _pre_tp_split(self, qkv3, gate, cos, sin, B, kv_mc, QS, b0=0, bg=None, outs=None):
        """ep2 gdn, QWEN36_ATTN_SPLIT=QS: pre_tp with the q and the gate heads spread over QS cores each (HS = NH / QS
        heads per core, kernels/attnp2_*.cpp): every core packs, norms and rotates its rows only (all row operations
        are per row, so the rows are bit-identical to pre_tp) and writes them; the last core of a role also writes
        the zero rows NH..31. Cores per sequence: QS q, 1 k, 1 v, QS gate."""
        m, NH, NKV = self.mesh, self.NH, self.NKV
        HS = NH // QS
        # t3c: b0 / bg / outs = only the sequences b0 .. b0 + bg - 1, into the given outputs (B > 16 groups)
        bg = B if bg is None else bg
        if outs is None:
            q_dec = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG)
            k_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
            v_sh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 32, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, kv_mc)
            gpk = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
        else:
            q_dec, k_sh, v_sh, gpk = outs
        kd, acc = self._kd, self._acc
        bs = range(b0, b0 + bg)
        # role -> jobs (b, h0, hs, last); the cores of a role are one contiguous index block (row rectangles)
        roles = (
            (0, qkv3, NH, self.wq, 0, q_dec, [(b, qi * HS, HS, qi == QS - 1) for b in bs for qi in range(QS)], 0),
            (1, qkv3, NKV, self.wk, 8 * NH, k_sh, [(b, 0, NKV, None) for b in bs], bg * QS),
            (2, qkv3, NKV, self.wk, 8 * NH + 8 * NKV, v_sh, [(b, 0, NKV, None) for b in bs], bg * QS + bg),
            (4, gate, NH, self.wq, 0, gpk, [(b, gi * HS, HS, gi == QS - 1) for b in bs for gi in range(QS)], bg * QS + 2 * bg),
        )
        kernels, comp = [], []
        for role, src, nh, w, base, dst, jobs, off in roles:
            rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k_, (b, h0, hs, last) in enumerate(jobs):
                c = self._core(off + k_)
                rr[c.x][c.y] = [src.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                                self.ones.buffer_address(), base, b, 2 * b, h0, hs]
                if last is None:  # k / v: whole tiles
                    rw[c.x][c.y] = [dst.buffer_address(), 8 * b, 0, 0, 0]
                else:
                    rw[c.x][c.y] = [dst.buffer_address(), 8 * b, h0, (16 - h0) if last else hs, int(last)]
            crs = self._span(off, len(jobs))
            # r64f: SRC_XT = the source's column tiles per row (one tile row of [1, 1, B, W]); the reader needs it
            # to address sequences >= 32 (rows 32..63 of the qkv3 / gate input)
            kernels.append(kd("attnp2_reader.cpp", crs,
                              [role, nh, 8, 0] + acc(src, w, cos, sin, self.ones) + [int(src.shape[-1]) // 32], rr,
                              ttnn.ReaderConfigDescriptor()))
            kernels.append(kd("attnp2_writer.cpp", crs, acc(dst), rw, ttnn.WriterConfigDescriptor()))
        kernels.append(kd(
            "attnp_compute.cpp", self._span(0, bg * QS + bg), self.ct, [],
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
        ))
        sizes = {0: (8, 0), 1: (8, 0), 2: (4, 0), 3: (1, 0), 5: (12, 0), 16: (8, 0), 24: (8, 1), 25: (1, 1), 26: (8, 1), 27: (8, 1)}
        io = [qkv3, gate, self.wq, self.wk, self.ones, cos, sin, q_dec, k_sh, v_sh, gpk]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                                   cbs=_cb(self._span(0, 2 * bg * QS + 2 * bg), sizes)))
        return q_dec, k_sh, v_sh, gpk

    def _post_tp_rows(self, attn, gpk, B, G, out):
        """ep2 gdn, QWEN36_ATTN_POST_ROWS=G: 8 x G cores, core (j, g) computes the products of its sequences only and
        writes their rows straight into the output tiles (attntpo3_*; the compute kernel as attntpo). Bit-identical."""
        NH = self.NH
        nper = -(-B // G)
        G = -(-B // nper)
        crs = self._span(0, 8 * G)
        rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(8 * G):
            c = self._core(i)
            j, g = i % 8, i // 8
            b0 = g * nper
            nb = min(nper, B - b0)
            rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address(), B, b0, nb]
            rw[c.x][c.y] = [j, out.buffer_address(), B, b0, nb, int(g == G - 1)]
            rc[c.x][c.y] = [nb]
        kd, acc = self._kd, self._acc
        kernels = [
            kd("attntpo3_reader.cpp", crs, acc(attn, gpk), rr, ttnn.ReaderConfigDescriptor()),
            kd("attntpo3_writer.cpp", crs, [NH] + acc(out), rw, ttnn.WriterConfigDescriptor()),
            kd("attntpo_compute.cpp", crs, [], rc,
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: (nper, 0), 1: (nper, 0), 3: (nper, 0), 16: (nper, 0)}
        ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=_cb(crs, sizes)))
        return out

    def post_tp(self, attn, gpk, B):
        NH = self.NH
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, NH * 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             ttnn.L1_MEMORY_CONFIG)
        # measured (micro-bench, 2 cards): B >= 2 -> one core per (column tile, sequence) writing rows directly
        # (B = 8: 15.5 -> 5.0 us, B = 2: 5.4 -> 3.8); B = 1 -> the head split below (3.6 vs 3.75). Default under
        # QWEN36_ATTN_SPLIT.
        G = int(os.environ.get("QWEN36_ATTN_POST_ROWS", str(B if B >= 2 else 0) if os.environ.get("QWEN36_ATTN_SPLIT") else "0"))
        if G >= 1:
            # t3c: at most the grid (8 cores per group of sequences; B 16: 8 groups of 2)
            return self._post_tp_rows(attn, gpk, B, min(G, B, (self.grid.x * self.grid.y) // 8), out)
        # ep2 gdn: QWEN36_ATTN_POST_SPLIT=HG -> HG cores per column tile j, each building the output tiles of NH / HG
        # heads (attntpo2_writer.cpp; the product tiles are computed on every core of j, the same ops: bit-identical)
        # measured (micro-bench, 2 cards, B 1 / 8): 3 cores per column tile best; default 3 under QWEN36_ATTN_SPLIT
        HG = int(os.environ.get("QWEN36_ATTN_POST_SPLIT", "3" if os.environ.get("QWEN36_ATTN_SPLIT") else "0"))
        HG = HG if HG >= 1 and NH % HG == 0 else 0  # 1 = the original work split on a row rectangle
        cores = [self._core(i) for i in range(8 * max(HG, 1))]
        crs = self._span(0, 8 * HG) if HG else ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            j, hg = i % 8, i // 8
            rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address(), B]
            rw[c.x][c.y] = [j, out.buffer_address(), B] + ([hg * (NH // HG), NH // HG] if HG else [])
            rc[c.x][c.y] = [B]
        kd, acc = self._kd, self._acc
        kernels = [
            kd("attntpo_reader.cpp", crs, acc(attn, gpk), rr, ttnn.ReaderConfigDescriptor()),
            kd("attntpo2_writer.cpp" if HG else "attntpo_writer.cpp", crs, [NH] + acc(out), rw, ttnn.WriterConfigDescriptor()),
            kd("attntpo_compute.cpp", crs, [], rc,
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: (B, 0), 1: (B, 0), 2: (NH, 0), 3: (B, 0), 16: (B, 0)}
        ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=_cb(crs, sizes)))
        return out
