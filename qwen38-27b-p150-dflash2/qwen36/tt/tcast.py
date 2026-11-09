# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 pf4 (QWEN36_PF_TCAST=1): bf16 -> fp32 typecast as one program over the whole grid (kernels/tcast_*.cpp)."""
import os

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))


def enabled():
    return os.environ.get("QWEN36_PF_TCAST", "0") == "1"


def to_fp32(x):
    mesh = x.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), ttnn.float32, ttnn.TILE_LAYOUT, mesh,
                                         ttnn.DRAM_MEMORY_CONFIG)
    import math
    nt = math.prod(list(x.padded_shape)) // 1024
    g = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i, j) for j in range(g.y) for i in range(g.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    per, rem = divmod(nt, len(cores))
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t = 0
    for i, c in enumerate(cores):
        k = per + (1 if i < rem else 0)
        rr[c.x][c.y] = [x.buffer_address(), t, t + k]
        rc[c.x][c.y] = [k]
        rw[c.x][c.y] = [out.buffer_address(), t, t + k]
        t += k

    def kd(src, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                     core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
    utd = [ttnn.UnpackToDestMode.Default] * 64
    utd[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = utd
    cbs = [ttnn.CBDescriptor(total_size=8 * ps, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ps)])
        for i, dt, ps in ((0, ttnn.bfloat16, 2048), (16, ttnn.float32, 4096))]
    acc = lambda t_: ttnn.TensorAccessorArgs(t_).get_compile_time_args()  # noqa: E731
    kernels = [kd("tcast_reader.cpp", acc(x), rr, ttnn.ReaderConfigDescriptor()),
               kd("tcast_writer.cpp", acc(out), rw, ttnn.WriterConfigDescriptor()),
               kd("tcast_compute.cpp", [], rc, cc)]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
