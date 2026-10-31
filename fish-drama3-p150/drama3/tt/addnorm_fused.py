# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-token residual add + RMSNorm as ONE program (port8, opt-in: DRAMA3_ADDNORM_FUSED=1).

Every decoder layer of the backbone runs ``h = x + attn; ff_in = norm(h); out = h + ff; next_in = norm(out)``, the
tower ``mixed = a + b; norm(mixed); h1 = mixed + att; norm(h1); s = d + h1; norm(s)``. Each add is one program and
each one-token norm three (tt/decode_norm.py: to width-sharded, LayerNorm on 8 cores, back). Here
(kernels/addnorm_*.cpp, port5's tt/rownorm_fused.py with a second input): every core of 8 packs the row-0 chunks of
x and d into 2 tiles each, adds them (bf16, as ttnn.add rounds), norms the sum and unpacks its 8 tiles of the sum
and of the normed row. Without d it is port5's one-program norm.
"""

import os
import struct

import torch

import ttnn

KDIR = "models/demos/blackhole/drama3/kernels"
ENABLED = os.environ.get("DRAMA3_ADDNORM_FUSED", "0") == "1"
# the residual sum's buffer: DRAM (as ttnn.add leaves it) or L1 interleaved
H_MC = ttnn.L1_MEMORY_CONFIG if os.environ.get("DRAMA3_ADDNORM_H_L1", "0") == "1" else ttnn.DRAM_MEMORY_CONFIG


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class AddNormFused:
    def __init__(self, mesh, dim=2048, n_cores=8):
        assert dim % (32 * 32) == 0 and (dim // 32) % n_cores == 0
        self.mesh, self.dim, self.nt = mesh, dim, dim // 32
        self.cores = [ttnn.CoreCoord(i, 0) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])
        self.per = self.nt // n_cores
        self.ones = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.wpk = {}

    def packed_weight(self, weight):
        """ROW_MAJOR [1, 1, dim / 32, 32] (row c = chunk c) -> TILE: row r of tile p = chunk 32 p + r. Cached; the
        first call of each norm runs eagerly in the warm-up, before any trace capture."""
        k = id(weight)
        if k not in self.wpk:
            self.wpk[k] = (weight, ttnn.to_layout(weight, ttnn.TILE_LAYOUT))
        return self.wpk[k][1]

    @staticmethod
    def _plain(t):
        if t.is_sharded():
            return ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG), True
        return t, False

    def __call__(self, x, d, weight, eps, out_mc=None, want_h=True):
        """x, d: one [.., 1, dim] TILE row each (d None: norm only) -> (h = x + d or None, norm(h) * w)."""
        w = self.packed_weight(weight)
        x, fx = self._plain(x)
        if d is not None:
            d, fd = self._plain(d)
        want_h = want_h and d is not None
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             out_mc or ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, H_MC) if want_h else out
        dd = d if d is not None else x
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rr[c.x][c.y] = [x.buffer_address(), dd.buffer_address(), w.buffer_address(), self.ones.buffer_address()]
            rw[c.x][c.y] = [i * self.per, out.buffer_address(), h.buffer_address()]
        np_ = self.nt // 32
        has_d = int(d is not None)
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("addnorm_reader.cpp", [self.nt, has_d] + acc(x, dd, w, self.ones), rr, ttnn.ReaderConfigDescriptor()),
            kd("addnorm_writer.cpp", [np_, self.per, int(want_h)] + acc(out, h), rw, ttnn.WriterConfigDescriptor()),
            kd("addnorm_compute.cpp", [np_, _bits(1.0 / self.dim), _bits(eps), has_d], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: np_, 1: np_, 2: 1, 3: (1 + has_d) * self.nt // 16, 4: (1 + int(want_h)) * self.per, 5: np_, 6: np_,
                 16: np_, 24: np_, 25: 1, 26: 1}
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in sizes.items()
        ]
        ios = [x, dd, w, self.ones, out] + ([h] if want_h else [])
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        if fx:
            ttnn.deallocate(x)
        if d is not None and fd:
            ttnn.deallocate(d)
        return (h if want_h else None), out


class AddNormFused2(AddNormFused):
    """v2 (DRAMA3_ADDNORM_V=2, kernels/addnorm2_*.cpp): core k reads, adds, norms and writes only its 8 tiles of the
    row; the 8 partial sums of squares are exchanged between the cores (32-byte writes + a semaphore)."""

    def __init__(self, mesh, dim=2048, n_cores=8):
        super().__init__(mesh, dim, n_cores)
        assert self.per <= 16 and n_cores <= 16
        self.noc = [mesh.worker_core_from_logical_core(c) for c in self.cores]

    def packed_weight(self, weight):
        """ROW_MAJOR [1, 1, dim / 32, 32] (row c = chunk c) -> TILE [1, 1, 32 n_cores, 32]: row i of tile k = chunk
        k * per + i, rows per..31 zero. Cached; built eagerly in the warm-up, before any trace capture."""
        k = id(weight)
        if k not in self.wpk:
            w = ttnn.to_torch(weight).reshape(self.nt, 32).to(torch.bfloat16)
            p = torch.zeros(len(self.cores), 32, 32, dtype=torch.bfloat16)
            p[:, : self.per] = w.reshape(len(self.cores), self.per, 32)
            self.wpk[k] = (weight, ttnn.from_torch(p.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh))
        return self.wpk[k][1]

    def __call__(self, x, d, weight, eps, out_mc=None, want_h=True):
        w = self.packed_weight(weight)
        x, fx = self._plain(x)
        if d is not None:
            d, fd = self._plain(d)
        want_h = want_h and d is not None
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             out_mc or ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, H_MC) if want_h else out
        dd = d if d is not None else x
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        nocs = [v for n in self.noc for v in (n.x, n.y)]
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rr[c.x][c.y] = [x.buffer_address(), dd.buffer_address(), w.buffer_address(), self.ones.buffer_address(), i]
            rw[c.x][c.y] = [i, out.buffer_address(), h.buffer_address()] + nocs
        has_d = int(d is not None)
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("addnorm2_reader.cpp", [self.per, has_d] + acc(x, dd, w, self.ones), rr, ttnn.ReaderConfigDescriptor()),
            kd("addnorm2_writer.cpp", [self.per, len(self.cores), int(want_h)] + acc(out, h), rw, ttnn.WriterConfigDescriptor()),
            kd("addnorm2_compute.cpp", [_bits(1.0 / self.dim), _bits(eps), has_d], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: 1, 1: 1, 2: 1, 3: 1, 4: (1 + int(want_h)) * self.per, 5: 1, 6: 1, 7: 1, 16: 1, 24: 1, 25: 1, 26: 1, 27: 1}
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in sizes.items()
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.crs, initial_value=0)]
        ios = [x, dd, w, self.ones, out] + ([h] if want_h else [])
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        if fx:
            ttnn.deallocate(x)
        if d is not None and fd:
            ttnn.deallocate(d)
        return (h if want_h else None), out


VERSION = int(os.environ.get("DRAMA3_ADDNORM_V", "2"))
_INST = {}


def get(mesh, dim):
    if dim not in _INST:
        _INST[dim] = (AddNormFused2 if VERSION == 2 else AddNormFused)(mesh, dim)
    return _INST[dim]
