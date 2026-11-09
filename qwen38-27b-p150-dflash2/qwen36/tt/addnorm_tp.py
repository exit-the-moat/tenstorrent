# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 (Exit the Moat, Episode 2): TP decode residual add + RMSNorm as ONE program (opt-in: QWEN36_ADDNORM_TP=1, needs
QWEN36_CCL_AR=1).

With QWEN36_CCL_AR the residual stream is replicated, width-sharded on the decode norm grid (res_mc: 32 cores x 5
tiles), and every norm runs as ttnn.add (sharded) -> sharded ttnn.rms_norm -> sharded-to-interleaved (the 1D decode
matmuls need an interleaved in0) -> matmul: three programs and their gaps. Here (kernels/addnt_*.cpp) each core adds
its x and d tiles (bf16, as ttnn.add rounds: h is the new residual, written into its res_mc shard), sums the squares
per user, exchanges the fp32 partials with the other cores (64-byte writes + a semaphore) and writes its tiles of
y = h * rsqrt(mean(h^2) + eps) * w straight to an interleaved output (L1 for the matmuls, DRAM for the LM head).
Not bit-identical to ttnn.rms_norm (sum of squares in another order, the product on the SFPU in fp32).

The layer loop hands the residual over unadded (Pending(h, d) = h + d): the last add of a layer and the next layer's
attention norm become one program, and so do the last layer's add and the final norm.
"""

import os
import struct

import torch

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))


_FPU = int(os.environ.get("QWEN36_ADDNORM_TP_FPU", "0"))  # final multiplies on the FPU (0: SFPU in fp32)
_FID = getattr(ttnn.MathFidelity, os.environ.get("QWEN36_ADDNORM_TP_FID", "HiFi2"))
_DBG = int(os.environ.get("QWEN36_ADDNORM_TP_DBG", "0"))  # timing experiments only (wrong values): 1 no exchange, 2 no final multiply


def enabled():
    return os.environ.get("QWEN36_ADDNORM_TP", "0") == "1"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class Pending:
    """A residual not added yet: the value is h + d (both res_mc)."""

    def __init__(self, h, d):
        self.h, self.d = h, d


def _start_end(cr):
    s = getattr(cr, "start", None) or getattr(cr, "start_coord")
    e = getattr(cr, "end", None) or getattr(cr, "end_coord")
    return s, e


class AddNormTP:
    def __init__(self, mesh, res_mc, dim):
        assert res_mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED, res_mc
        spec = res_mc.shard_spec
        assert spec.orientation == ttnn.ShardOrientation.ROW_MAJOR and spec.shape[0] == 32, spec
        s, e = _start_end(spec.grid.bounding_box())
        cores = [ttnn.CoreCoord(x, y) for y in range(s.y, e.y + 1) for x in range(s.x, e.x + 1)]
        assert len(cores) == spec.grid.num_cores(), (len(cores), spec.grid.num_cores())
        self.per = spec.shape[1] // 32
        assert len(cores) * self.per * 32 == dim, (len(cores), self.per, dim)
        self.mesh, self.res_mc, self.dim, self.cores = mesh, res_mc, dim, cores
        self.nc = len(cores)
        self.crs = spec.grid
        self.noc = [mesh.worker_core_from_logical_core(c) for c in cores]
        self.ones = ttnn.from_torch(
            torch.ones(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.wpk = {}

    def packed_weight(self, weight):
        """RMSNorm weight ROW_MAJOR [1, 1, dim / 32, 32] (+1 already applied) -> TILE [1, 1, 32, dim] (every row the
        weight), interleaved DRAM (10 KB per core and norm in L1 would push the persistent L1 into the prefill CB
        region: 4 layers already clashed). Before any trace capture."""
        k = id(weight)
        if k not in self.wpk:
            w = ttnn.to_torch(ttnn.get_device_tensors(weight)[0]).reshape(-1)[: self.dim].to(torch.bfloat16)
            t = w.reshape(1, 1, 1, self.dim).repeat(1, 1, 32, 1).contiguous()
            self.wpk[k] = (
                weight,
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
            )
        return self.wpk[k][1]

    def fits(self, x):
        return (x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and x.is_sharded()
                and x.memory_config() == self.res_mc and x.shape[-1] == self.dim and x.shape[-2] <= 32)

    def __call__(self, x, d, norm, out_mc=ttnn.L1_MEMORY_CONFIG):
        """x, d: [1, 1, B, dim] bf16 in res_mc (d None: norm only); norm: the models.common RMSNorm (weight, eps).
        -> (h = x + d in res_mc, or None; y = norm(h) in out_mc, interleaved)."""
        assert self.fits(x) and (d is None or self.fits(d)), (x.memory_config(), None if d is None else d.memory_config())
        w = self.packed_weight(norm.weight)
        has_d = int(d is not None)
        shape = ttnn.Shape(list(x.shape))
        y = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        h = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, self.res_mc) if has_d else None
        dd = d if has_d else x
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        nocs = [v for n in self.noc for v in (n.x, n.y)]
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rr[c.x][c.y] = [x.buffer_address(), dd.buffer_address(), w.buffer_address(), self.ones.buffer_address(), i]
            rw[c.x][c.y] = [i, y.buffer_address(), h.buffer_address() if has_d else 0] + nocs

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
            math_fidelity=_FID, fp32_dest_acc_en=True, math_approx_mode=False
        )
        utd = [ttnn.UnpackToDestMode.Default] * 64
        P = self.per
        NG = (self.nc + 31) // 32
        # CB index -> (tiles, fp32?, unpack to dest in fp32?)
        spec = {0: (P, 0, 0), 1: (P, 0, 0), 2: (P, 0, 0), 3: (1, 0, 0), 4: (P, 0, 0), 5: (P, 1, 0), 6: (1, 1, 0),
                7: (NG, 1, 1), 8: (NG, 1, 0), 9: (1, 1, 0 if _FPU else 1), 16: (P, 0, 0)}
        cbs = []
        for i, (nt, f32, u) in spec.items():
            if u:
                utd[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ps = 4096 if f32 else 2048
            cbs.append(ttnn.CBDescriptor(
                total_size=ps * nt, core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32 if f32 else ttnn.bfloat16, page_size=ps)],
            ))
        cc.unpack_to_dest_mode = utd
        kernels = [
            kd("addnt_reader.cpp", [P, has_d] + acc(self.ones, w), rr, ttnn.ReaderConfigDescriptor()),
            kd("addnt_writer.cpp", [P, self.nc, has_d, _DBG] + acc(y), rw, ttnn.WriterConfigDescriptor()),
            kd("addnt_compute.cpp", [P, _bits(1.0 / self.dim), _bits(norm.eps), has_d, _DBG, _FPU, NG], [], cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.crs, initial_value=0)]
        ios = [x, dd, w, self.ones, y] + ([h] if has_d else [])
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return h, y


_INST = {}


def get(mesh, res_mc, dim):
    k = (str(res_mc), dim)
    if k not in _INST:
        _INST[k] = AddNormTP(mesh, res_mc, dim)
    return _INST[k]
