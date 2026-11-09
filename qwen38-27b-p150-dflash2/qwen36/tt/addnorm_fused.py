# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 (Exit the Moat, Episode 2): one-token residual add + RMSNorm as ONE program (opt-in: QWEN36_ADDNORM_FUSED=1).

Single-device decode (the 1-card path) runs per layer ``a = norm(x); h = x + attn; f = norm(h); out = h + ff``; each
norm is ttnn.rms_norm on an interleaved [1, 1, 5120] row, which runs on one core (~81 us, BREAKDOWN.md row 4), plus a
to_memory_config to L1. Here (kernels/addn_*.cpp, port of the Episode 0 Drama 3 AddNormFused2) NC cores each pack
their PER row-0 chunks of x (and d) into one tile, add (bf16, as ttnn.add rounds), square, exchange the fp32 partial
sums (32-byte writes + a semaphore) and write their tiles of h and of y = h * rsqrt(mean(h^2) + eps) * w.
Without d it is a plain one-program norm. Not bit-identical to ttnn.rms_norm (fp32 sum of squares in a different
order, the product on the SFPU in fp32): h is bit-equal to ttnn.add.
"""

import math
import os
import struct

import torch

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))


def enabled():
    return os.environ.get("QWEN36_ADDNORM_FUSED", "0") == "1"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# CB index -> (tiles, fp32?, unpack to dest?)
def _cbs(per, want_h):
    return {0: (1, 0, 0), 1: (1, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: ((1 + int(want_h)) * per, 0, 0), 5: (1, 0, 0),
            6: (1, 0, 0), 7: (1, 1, 0), 16: (1, 0, 0), 24: (1, 1, 0), 25: (1, 1, 0), 26: (1, 1, 1), 27: (1, 1, 0)}


class AddNorm:
    """For one row width (dim = 32 * NC * PER); norm weights packed per RMSNorm module on first use (eager warm-up)."""

    def __init__(self, mesh, dim=5120, n_cores=10):
        nt = dim // 32
        assert dim % 32 == 0 and nt % n_cores == 0 and nt // n_cores <= 16 and n_cores <= 16, (dim, n_cores)
        self.mesh, self.dim, self.nt, self.nc, self.per = mesh, dim, nt, n_cores, nt // n_cores
        self.cores = [ttnn.CoreCoord(i, 0) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])
        self.noc = [mesh.worker_core_from_logical_core(c) for c in self.cores]
        self.ones = ttnn.from_torch(
            torch.ones(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        self.wpk = {}

    def packed_weight(self, weight):
        """RMSNorm weight ROW_MAJOR [1, 1, dim / 32, 32] (row c = chunk c, +1 already applied) -> TILE
        [1, 1, 32 NC, 32]: row i of tile k = chunk k * PER + i, rows PER..31 zero."""
        k = id(weight)
        if k not in self.wpk:
            w = ttnn.to_torch(weight).reshape(self.nt, 32).to(torch.bfloat16)
            p = torch.zeros(self.nc, 32, 32, dtype=torch.bfloat16)
            p[:, : self.per] = w.reshape(self.nc, self.per, 32)
            self.wpk[k] = (
                weight,
                ttnn.from_torch(p.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh),
            )
        return self.wpk[k][1]

    def fits(self, t):
        """One bf16 TILE interleaved row of this width (the kernel's contract)."""
        return (
            t.dtype == ttnn.bfloat16
            and t.layout == ttnn.TILE_LAYOUT
            and not t.is_sharded()
            and t.shape[-1] == self.dim
            and math.prod(list(t.shape)) == self.dim
        )

    def __call__(self, x, d, norm, out_mc=ttnn.L1_MEMORY_CONFIG, h_mc=None):
        """x, d: one bf16 TILE row [.., 1, dim] each, interleaved (d None: norm only); norm: the models.common RMSNorm
        (weight, eps). -> (h = x + d or None, y = norm(h)), both [1, 1, dim]."""
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded(), (x.dtype, x.layout)
        w = self.packed_weight(norm.weight)
        want_h = d is not None
        shape = ttnn.Shape(list(x.shape))
        assert x.shape[-1] == self.dim and math.prod(list(x.shape)) == self.dim, x.shape
        out = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        h = (
            ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, h_mc or x.memory_config())
            if want_h
            else out
        )
        dd = d if d is not None else x
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        nocs = [v for n in self.noc for v in (n.x, n.y)]
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rr[c.x][c.y] = [x.buffer_address(), dd.buffer_address(), w.buffer_address(), self.ones.buffer_address(), i]
            rw[c.x][c.y] = [i, out.buffer_address(), h.buffer_address()] + nocs
        has_d = int(d is not None)

        def kd(src, ct, rt, cfg):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/{src}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )

        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        utd = [ttnn.UnpackToDestMode.Default] * 64
        cbs = []
        for i, (nt, f32, u) in _cbs(self.per, want_h).items():
            if u:
                utd[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ps = 4096 if f32 else 2048
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=ps * nt,
                    core_ranges=self.crs,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=i, data_format=ttnn.float32 if f32 else ttnn.bfloat16, page_size=ps
                        )
                    ],
                )
            )
        cc.unpack_to_dest_mode = utd
        kernels = [
            kd("addn_reader.cpp", [self.per, has_d] + acc(x, dd, w, self.ones), rr, ttnn.ReaderConfigDescriptor()),
            kd("addn_writer.cpp", [self.per, self.nc, int(want_h)] + acc(out, h), rw, ttnn.WriterConfigDescriptor()),
            kd("addn_compute.cpp", [_bits(1.0 / self.dim), _bits(norm.eps), has_d], [], cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.crs, initial_value=0)]
        ios = [x, dd, w, self.ones, out] + ([h] if want_h else [])
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return (h if want_h else None), out


_INST = {}


def get(mesh, dim):
    if dim not in _INST:
        _INST[dim] = AddNorm(mesh, dim)
    return _INST[dim]
