# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-token RMSNorm as ONE program (port5 Track B, opt-in: DRAMA3_ROWNORM_FUSED=1).

tt/decode_norm.py's ShardedRowNorm (to width-sharded, LayerNorm on 8 cores, back to interleaved: 3 programs,
7.4 us, tests/profile_ops_dev.py) runs ~145 times per frame (backbone 81, tower 65). Here (kernels/rownorm_*.cpp):
every core of 8 packs the 64 row chunks of the [1, 1, 2048] row into 2 tiles, computes the norm on them (the sum of
squares via two all-ones matmuls), and unpacks its 8 output tiles; same values as x * rsqrt(mean(x^2) + eps) * w.
"""

import os
import struct

import torch

import ttnn

KDIR = "models/demos/blackhole/drama3/kernels"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class RowNormFused:
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

    def __call__(self, x, weight, eps, out_mc=None):
        w = self.packed_weight(weight)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             out_mc or ttnn.DRAM_MEMORY_CONFIG)
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rr[c.x][c.y] = [x.buffer_address(), w.buffer_address(), self.ones.buffer_address()]
            rw[c.x][c.y] = [i * self.per, out.buffer_address()]
        np_ = self.nt // 32
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("rownorm_reader.cpp", [self.nt] + acc(x, w, self.ones), rr, ttnn.ReaderConfigDescriptor()),
            kd("rownorm_writer.cpp", [np_, self.per] + acc(out), rw, ttnn.WriterConfigDescriptor()),
            kd("rownorm_compute.cpp", [np_, _bits(1.0 / self.dim), _bits(eps)], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: np_, 1: np_, 2: 1, 3: self.nt // 16, 4: self.per, 16: np_, 24: np_, 25: 1, 26: 1}
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in sizes.items()
        ]
        ttnn.generic_op([x, w, self.ones, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


_FUSED = {}


def install(mesh):
    """Patch decode_norm.ShardedRowNorm (every instance: backbone and tower) to the one-program norm."""
    if os.environ.get("DRAMA3_ROWNORM_FUSED", "0") != "1" or _FUSED:
        return bool(_FUSED)
    from models.demos.blackhole.drama3.tt import decode_norm

    orig = decode_norm.ShardedRowNorm.__call__

    def call(self, x, weight, eps, ckc, out_mc=None):
        dim = x.shape[-1]
        if x.shape[-2] != 1 or dim % 1024 or len(x.shape) != 3:
            return orig(self, x, weight, eps, ckc, out_mc)
        if dim not in _FUSED:
            _FUSED[dim] = RowNormFused(mesh, dim)
        return _FUSED[dim](x, weight, eps, out_mc)

    decode_norm.ShardedRowNorm.__call__ = call
    _FUSED[0] = None  # installed
    return True
