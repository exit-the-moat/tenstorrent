# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 pf4 (QWEN36_GDN_FUSE2=1): the GDN prefill output gate as one program (kernels/gatek_*.cpp).

Replaces nlp_concat_heads (head-major normed o -> token-major), the z slice of the fused projection output, ttnn.silu
and ttnn.multiply: gated[r, c] = n[h, r, d] * silu(z[r, c]) with h, d from c, z read in place from qkvzab. The math
is the stock ops' (n fp32: see kernels/gatek_compute.cpp), the rest is indexing: bit-identical (pf4/fuse2_test.py).
"""
import os

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"))


def enabled():
    return os.environ.get("QWEN36_GDN_FUSE2", "0") == "1"


def gate(mesh, n, qkvzab, Nv, Dv, z_col):
    """n: rms-normed o, head-major [1, Nv, T, Dv] TILE; qkvzab [1, T, NQ*32] TILE; z = qkvzab[..., z_col:z_col+Nv*Dv].
    -> gated [1, T, Nv*Dv] TILE DRAM."""
    T = n.shape[-2]
    NR, DT, NC = T // 32, Dv // 32, Nv * Dv // 32
    NQ = -(-qkvzab.shape[-1] // 32)  # tiles per row (logical width 8240 is padded to 8256)
    assert z_col % 32 == 0 and T % 32 == 0 and Dv % 32 == 0 and z_col + Nv * Dv <= qkvzab.shape[-1]
    assert n.dtype == ttnn.float32 and qkvzab.dtype == ttnn.bfloat16, (n.dtype, qkvzab.dtype)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, Nv * Dv]), ttnn.float32, ttnn.TILE_LAYOUT, mesh,
                                         ttnn.DRAM_MEMORY_CONFIG)
    g = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    total = NR * NC
    per, rem = divmod(total, len(cores))
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t = 0
    for i, c in enumerate(cores):
        k = per + (1 if i < rem else 0)
        rr[c.x][c.y] = [n.buffer_address(), qkvzab.buffer_address(), t, t + k, NR]
        rc[c.x][c.y] = [k]
        rw[c.x][c.y] = [out.buffer_address(), t, t + k]
        t += k
    acc = lambda *ts: [a for x in ts for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]  # noqa: E731

    def kd(src, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                     core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
    utd = [ttnn.UnpackToDestMode.Default] * 64
    utd[0] = utd[1] = ttnn.UnpackToDestMode.UnpackToDestFp32  # as binary_ng's SFPU path
    cc.unpack_to_dest_mode = utd
    cbs = [ttnn.CBDescriptor(total_size=8 * ps, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ps)])
        for i, dt, ps in ((0, ttnn.float32, 4096), (1, ttnn.bfloat16, 2048), (16, ttnn.float32, 4096))]
    kernels = [
        kd("gatek_reader.cpp", [NC, DT, NQ, z_col // 32] + acc(n, qkvzab), rr, ttnn.ReaderConfigDescriptor()),
        kd("gatek_writer.cpp", acc(out), rw, ttnn.WriterConfigDescriptor()),
        kd("gatek_compute.cpp", [], rc, cc),
    ]
    ttnn.generic_op([n, qkvzab, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
