# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ms4 (Exit the Moat, Episode 2, 12 Nov): the DFlash2 verify GDN chain in block (WY) form, DF_GDN_WY=1 (default off).

kernels/gdnwy_{reader,writer,compute}.cpp: one program per GDN layer for every live stream of a k = 8 verifier (a core
group per stream, CS cores per value head). Instead of 8 chained slots that each write their state (bufs[0..7]), the
block is solved in closed form from S0, and only S0 (-> base) and the block's K / U / R / gamma (-> blk, 10 fp32 tiles
per value head) are stored; the next step recomputes the state at its accepted slot from them. Needs the preset's
DF_GDN_PRO=1 (prebuilt tap tiles), QWEN36_GDN_FAST >= 4 (CS), fp32 state.

  multi[s] = dict(Xl, base, blk, kvm, xrow0, orow0, oz0)  (as FusedGDNDecodeTP.chain_multi, blk instead of bufs)
"""
import os

import torch

import ttnn

from models.demos.blackhole.qwen36.tt.gdn.fused_decode import KDIR, BF16_TILE, FP32_TILE

# CB index -> (tiles (int, or "NC" / "4NC" / "NT9"), fp32, unpack to dest)
_CBS = {
    3: ("NC", 0, 0), 4: (1, 0, 0), 5: (1, 0, 0), 6: (1, 0, 0), 7: (1, 0, 0), 9: ("NC", 0, 0), 10: (1, 0, 0),
    16: ("NC", 0, 0), 33: (1, 0, 0), 37: (8, 0, 0), 38: (8, 0, 0), 41: ("NT9", 0, 0),
    8: ("4NC", 1, 1), 14: ("NC", 1, 1), 17: ("4NC", 1, 0), 21: ("4NC", 1, 0), 24: ("NC", 1, 0), 25: ("NC", 1, 1),
    26: ("NC", 1, 0), 28: ("NC", 1, 0), 29: (1, 1, 0), 35: (1, 1, 0), 36: (1, 1, 0), 39: (1, 1, 0), 40: (1, 1, 0),
    42: (4, 1, 0), 43: (4, 1, 0), 46: (4, 1, 0), 47: (1, 1, 1), 48: (1, 1, 1), 50: (4, 1, 0), 51: ("NC", 1, 0),
    52: (1, 1, 0), 53: (1, 1, 1), 54: (4, 1, 0), 55: ("NC", 1, 0), 56: (1, 1, 1), 57: (1, 1, 1), 58: (1, 1, 0),
    59: (1, 1, 0), 60: ("NC", 1, 0), 61: (1, 1, 0), 62: (4, 1, 1), 63: (1, 1, 1), 0: (1, 1, 1), 1: (1, 1, 1),
    2: (1, 1, 0), 11: (1, 1, 1), 12: (1, 1, 0), 13: (1, 1, 0), 15: (1, 1, 0), 18: (1, 1, 0), 19: (1, 1, 1),
    20: (1, 1, 1),
}


KSUF = ""  # kernel file suffix (dev iterations in a live server: a new file name = a fresh JIT build)


def enabled():
    return os.environ.get("DF_GDN_WY", "0") == "1"


def alloc_blk(fd, mesh):
    """One stream's per-layer block store: [1, Nv * 10 * 32, 32] fp32 TILE (page hv * 10 + t), per device."""
    return ttnn.from_torch(torch.zeros(1, fd.Nv * 10 * 32, 32, dtype=torch.float32), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def _program(fd, qkvzab, out, multi, NT, commit=False):
    Nv, Nk = fd.Nv, fd.Nk
    ZT0, ACOL, BCOL, CMODE, STF32 = fd.layout
    assert STF32 == 1 and CMODE == 1, (STF32, CMODE)
    assert getattr(fd, "wbat", None) is not None, "DF_GDN_WY needs DF_GDN_PRO=1 at model build (prebuilt tap tiles)"
    npg = len(multi)
    CS = fd._cs(npg)
    assert CS in (1, 2, 4), CS
    NC = 4 // CS
    n = npg * Nv * CS
    grid = fd.grid
    assert n <= grid.x * grid.y, (n, grid)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
    noc = [fd.mesh.worker_core_from_logical_core(c) for c in cores]
    crs = ttnn.num_cores_to_corerangeset(n, grid, True)
    m0 = multi[0]
    xq = qkvzab if qkvzab is not None else m0["base"]
    o_ = out if out is not None else m0["base"]
    xl0 = m0["Xl"] if m0.get("Xl") is not None else m0["base"]
    rct = [Nk, Nv, ZT0, CS]
    for t in (xq, xl0, fd.wbat, m0["base"], fd.onw, m0["kvm"], m0["blk"]):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [Nv, CS, 1, ACOL, BCOL]
    for t in (m0["base"], o_, xq, fd.dtbf, fd.anegf, m0["blk"]):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        pl, hv, ci = i // (Nv * CS), (i // CS) % Nv, i % CS
        m_ = multi[pl]
        xl = m_["Xl"] if m_.get("Xl") is not None else m_["base"]
        peers = []
        for cj in range(CS):
            if cj != ci:
                pc = noc[(pl * Nv + hv) * CS + cj]
                peers += [pc.x, pc.y]
        rrt[c.x][c.y] = [hv, ci, xq.buffer_address(), xl.buffer_address(), fd.wbat.buffer_address(),
                         m_["base"].buffer_address(), fd.onw.buffer_address(), m_["kvm"].buffer_address(),
                         m_["blk"].buffer_address(), int(m_.get("xrow0", 0))]
        wrt[c.x][c.y] = [hv, ci, o_.buffer_address(), xq.buffer_address(), fd.dtbf.buffer_address(),
                         fd.anegf.buffer_address(), m_["base"].buffer_address(), m_["blk"].buffer_address(),
                         int(m_.get("xrow0", 0)), int(m_.get("orow0", 0)), int(m_.get("oz0", 32))] + peers
        crt[c.x][c.y] = [ci * NC]
    defs = [("GDNQ_CHAIN", str(int(NT)))] + ([("GDNWY_COMMIT", "1")] if commit else [])

    def kd(src, ct, rt, cfg):
        k = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                  core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)
        k.defines = list(k.defines) + defs
        return k

    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                      math_approx_mode=False)
    utd = [ttnn.UnpackToDestMode.Default] * 64
    cbs = []
    sz = {"NC": NC, "4NC": 4 * NC, "NT9": 8 + NC}
    for idx, (nt, f32, u) in _CBS.items():
        nt = sz.get(nt, nt)
        if u:
            utd[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ps = FP32_TILE if f32 else BF16_TILE
        fmt = ttnn.float32 if f32 else ttnn.bfloat16
        cbs.append(ttnn.CBDescriptor(total_size=ps * nt, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=ps)]))
    cc.unpack_to_dest_mode = utd
    kernels = [kd(f"gdnwy_reader{KSUF}.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
               kd(f"gdnwy_writer{KSUF}.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
               kd(f"gdnwy_compute{KSUF}.cpp", list(fd.ct) + [CS], crt, cc)]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0)]
    io = [xq, fd.wbat, fd.dtbf, fd.anegf, fd.onw]
    for m_ in multi:
        io += [m_["Xl"] if m_.get("Xl") is not None else m_["base"], m_["base"], m_["kvm"], m_["blk"]]
    io += [o_]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), io


def _cached(fd, key, build):
    pc = fd.__dict__.setdefault("_pcache_wy", {})
    ck = key[:3]
    hit = pc.get(ck)
    if hit is None or hit[0] != key:
        hit = (key,) + build()
        pc[ck] = hit
    return hit


def chain_wy(fd, qkvzab, out, multi, NT):
    """The WY verify of all streams in `multi` (one core group each): gated rows -> out, S0 -> base, block -> blk."""
    key = ("v" + KSUF, len(multi), int(multi[0]["xrow0"]), int(NT), qkvzab.buffer_address(), out.buffer_address())
    for m_ in multi:
        key += tuple(t.buffer_address() for t in (m_["Xl"], m_["base"], m_["kvm"], m_["blk"]))
        key += (int(m_["xrow0"]), int(m_["orow0"]), int(m_["oz0"]))
    hit = _cached(fd, key, lambda: _program(fd, qkvzab, out, multi, NT))
    io = [qkvzab, fd.wbat, fd.dtbf, fd.anegf, fd.onw]
    for m_ in multi:
        io += [m_["Xl"], m_["base"], m_["kvm"], m_["blk"]]
    io += [out]
    assert len(io) == len(hit[2])
    return ttnn.generic_op(io, hit[1])


def commit(fd, base, blk, kvm, NT):
    """S0 (the state at the accepted slot a in kvm, recomputed from base + blk) -> base. a >= NT: no change."""
    m_ = dict(base=base, blk=blk, kvm=kvm)
    key = ("c" + KSUF, 1, 0, int(NT), base.buffer_address(), blk.buffer_address(), kvm.buffer_address())
    hit = _cached(fd, key, lambda: _program(fd, None, None, [m_], NT, commit=True))
    io = [base, fd.wbat, fd.dtbf, fd.anegf, fd.onw, base, base, kvm, blk, base]
    assert len(io) == len(hit[2])
    return ttnn.generic_op(io, hit[1])
