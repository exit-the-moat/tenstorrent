# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ep2 (Exit the Moat, Episode 2): fused one-token GDN decode (opt-in: QWEN36_GDN_FUSED_DECODE=1), for the 2-card
TPGatedDeltaNet (FusedGDNDecodeTP) and the 1-card Qwen36GatedDeltaNet (FusedGDNDecode1, B = 1, bf16 state).

TPGatedDeltaNet.forward_decode runs ~45 small ttnn ops between the qkvzab in-projection and the out projection (5
slices, the conv shift register (4 copies + mul + 3 mac + silu), q/k/v slices and reshapes, 2 repeat_interleave, the
gates, the fp32 delta-rule recurrence with the state moved to L1 and back, the gated rms norm). Here they are ONE
program (kernels/gdnq_*.cpp via ttnn.generic_op): one core per (value head, row pair), conv state and fp32 recurrent
state updated in place. Derived from the Episode 0 Drama 3 kernel (drama3/kernels/gdn_decode3_*).

Not bit-identical to the op chain: the conv sums stay fp32 until the silu (the chain rounds each mac to bf16), the
gates are fp32 (the chain rounds a + dt_bias and g to bf16), the state math is fp32 on the SFPU, the matmul operands
are tf32 (as in the chain). Gate it like any numerics change.
"""

import os
import struct

import torch

import ttnn

# absolute path next to this package (works whatever the mount layout / TT_METAL_HOME)
KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"))
BF16_TILE, FP32_TILE = 2048, 4096
# CB index -> (tiles, fp32?, unpack to dest?)
CBS = {
    0: (2, 0, 0), 1: (6, 0, 0), 2: (4, 0, 0), 3: (8, 0, 0), 4: (2, 0, 0), 5: (2, 0, 0), 6: (1, 0, 0), 7: (1, 0, 0),
    8: (16, 1, 1), 9: (4, 0, 0), 10: (1, 0, 0), 11: (12, 0, 0), 12: (3, 0, 0), 13: (16, 1, 0), 14: (4, 1, 1),
    15: (1, 0, 0), 16: (4, 0, 0), 17: (16, 1, 0), 18: (1, 1, 0), 19: (12, 0, 0), 20: (1, 1, 0), 21: (4, 1, 0),
    22: (4, 1, 0), 23: (4, 1, 0), 24: (4, 1, 0), 25: (4, 1, 1), 26: (4, 1, 0), 28: (4, 1, 0), 29: (1, 1, 0),
    30: (1, 1, 1), 31: (1, 1, 1),
}


def enabled():
    return os.environ.get("QWEN36_GDN_FUSED_DECODE", "0") == "1"


def fast():
    """ep2 gdn: QWEN36_GDN_FAST=1 -> kernels/gdnq2_*.cpp (all independent reads up front, CS cores per value head);
    3 -> gdnq3_reader / gdnq3_compute + gdnq2_writer (packed rows of both sequences read straight from DRAM, selectors
    made in L1, reads phased by first use, CS cores per value head); 4 -> gdnq4_* (v3 + the q / k conv rows read once
    per key head and forwarded L1 -> L1, the gate inputs from the writer RISC, gates first); 5 -> v4 with the compute
    gdnq5_compute (up to 4 tiles per DST session, q / k norms side by side; no gain, wrong with 2 sequences per core:
    do not use); 6 -> gdnq6_reader + gdnq4 writer / compute (v4 + with B > 2 the conv rows of all row pairs read once
    per channel tile by the pair-0 core and handed out L1 -> L1); 7 -> gdnq7_reader + gdnq4 writer / compute (v6 with
    the two state CBs (matmul copy 13, SFPU copy 8) on ONE allocation: no 64 KB L1 copy, the states of both sequences
    of the pair read up front in the same footprint; exact, slower at B = 8: record only); 8 -> gdnq8_reader +
    gdnq8_writer + gdnq4 compute (v6 at every B, the first state read by the writer RISC right after the gate inputs,
    in parallel with the conv rows on the reader). 0 = v1."""
    return int(os.environ.get("QWEN36_GDN_FAST", "0"))


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class _FusedGDNBase:
    """Constants and the program for one GDN layer (built after its weights exist, before any trace capture).

    Subclasses set mesh, Nk, Nv, layout = (ZT0, ACOL, BCOL, CMODE, STF32) and implement _conv() (st0..st3: the conv
    tensors the reader reads / writes; CMODE 0: st1..st3 = the 3 history rows; 1: + st0 = where the oldest row goes;
    2: st1 alone holds the history in rows 0..2) and _rec() (the recurrent state [Bmax, Nv, 128, 128])."""

    def _build(self, taps, dtb, aneg, onw, qkv_dim, norm_eps):
        """taps: [tap][dev] torch [qkv_dim]; dtb / aneg: [dev] torch [Nv]; onw: torch [128] (same on every device)."""
        mesh, Nk, Nv = self.mesh, self.Nk, self.Nv
        nd = mesh.get_num_devices()

        def up(per_dev):  # list (one per device) of torch tensors -> mesh tensor (shard d on device d), bf16 TILE DRAM
            full = torch.cat([p.unsqueeze(0) for p in per_dev], dim=0).to(torch.bfloat16)
            return ttnn.from_torch(
                full,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if nd > 1 else None,
            )

        wpk, wpk3, dtbf, anegf = [], [], [], []
        for d in range(nd):
            tp = [taps[j][d].flatten()[:qkv_dim].reshape(-1, 32) for j in range(4)]  # rows = channel tiles
            w = torch.zeros(Nv * 4 * 32, 32)
            for hv in range(Nv):
                hk = hv // (Nv // Nk)
                ch = [4 * hk + t for t in range(4)] + [4 * Nk + 4 * hk + t for t in range(4)]
                ch += [8 * Nk + 4 * hv + t for t in range(4)]
                for j in range(4):
                    w[(hv * 4 + j) * 32 : (hv * 4 + j) * 32 + 12] = tp[j][ch]
            wpk.append(w)
            # v3: the same taps at packed rows R(t, s) = 16 (t // 8) + 2 (t % 8) + s (both sequences of a row pair)
            w3 = torch.zeros(Nv * 4 * 32, 32)
            for hv in range(Nv):
                for j in range(4):
                    for t in range(12):
                        for sq in range(2):
                            w3[(hv * 4 + j) * 32 + 16 * (t // 8) + 2 * (t % 8) + sq] = w[(hv * 4 + j) * 32 + t]
            wpk3.append(w3)
            dtbf.append(dtb[d].flatten()[:Nv].repeat_interleave(32 * 32).reshape(Nv * 32, 32))
            anegf.append(aneg[d].flatten()[:Nv].repeat_interleave(32 * 32).reshape(Nv * 32, 32))
        self.wpk = up(wpk)
        # ep2 t3b DF_GDN_PRO: the taps as tiles with every row = the taps of one channel tile, page (hv*4 + j)*12 + t
        self.wbat = None
        if os.environ.get("DF_GDN_PRO", "0") == "1":
            wb = []
            for d in range(nd):
                w_ = wpk[d].reshape(Nv * 4, 32, 32)[:, :12, :]  # [hv*4 + j, t, 32]
                wb.append(w_.unsqueeze(2).expand(-1, -1, 32, -1).reshape(Nv * 4 * 12 * 32, 32))
            self.wbat = up(wb)
        self.wpk3 = up(wpk3) if fast() >= 3 else None
        self.dtbf = up(dtbf)
        self.anegf = up(anegf)
        o = torch.zeros(32, 128)
        o[0] = onw.flatten()[:128]
        self.onw = up([o] * nd)
        sel = torch.zeros(12 * 32, 32)
        sel[torch.arange(12) * 32, torch.arange(12)] = 1.0
        self.sel = up([sel] * nd)
        self.ct = [_bits(128**-0.5), _bits(1e-6), _bits(1.0), _bits(1.0 / 128), _bits(norm_eps), _bits(20.0)]
        self.grid = mesh.compute_with_storage_grid_size()
        self.fast = fast() if type(self).__name__ == "FusedGDNDecodeTP" else 0

    def ok(self, x):
        """The kernel reads bf16 TILE rows; the in-projection output takes x's dtype."""
        return x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT

    def _cs(self, npairs):
        """v2: cores per (value head, row pair), 4 / 2 / 1 state column tiles each; QWEN36_GDN_CS caps it."""
        if not getattr(self, "fast", False):
            return 0
        # measured (micro-bench, 2 cards, row-rectangle core ranges): 4 is best at every B (B = 1: 30.1 us vs 35.8 at 2)
        cs = int(os.environ.get("QWEN36_GDN_CS", "4"))
        while cs > 1 and npairs * self.Nv * cs > self.grid.x * self.grid.y:
            cs //= 2
        return cs

    def _program(self, qkvzab, B, out, chain=None, chainall=None, group=None, multi=None):
        """chain (ep2 t3, DFlash2 chained verify): (xr, rec_in, rec_out, last) = read the x / z / gate row xr of a
        [1, T, W] qkvzab, the state from rec_in row 0, write the state to rec_out row 0 and the gated row into row xr of
        `out` ([1, T, Nv*128]); `last` = this call zeroes the output tile rows below xr (rows xr + 1 always). B = 1."""
        Nv = self.Nv
        ZT0, ACOL, BCOL, CMODE, STF32 = self.layout
        assert CMODE != 2 or B == 1, "one-tensor conv history: B = 1 only"
        npairs = (B + 1) // 2
        self._B = B
        # t3c group = (P0, NPG): this program covers the row pairs P0 .. P0 + NPG - 1 only (B > 8: one program per group
        # of pairs, so the cores fit the grid); the pair index p stays absolute (rows, state rows, last), the core
        # lists are local (pl = p - P0)
        P0, npg = group if group is not None else (0, npairs)
        if multi is not None:
            # ms1o (multi-stream DFlash2 verify, GDNQ_MSRT): stream s = core group s (Nv * CS cores, CS reduced until
            # all groups fit the grid), each group its own stream's window / base / bufs / `a` and its rows of the
            # shared in-projection and output (runtime args); every core computes B = 1, exactly as chain_all
            assert chainall is not None and B == 1
            P0, npg = 0, len(multi)
        CS = self._cs(npg)  # 0 = v1 kernels
        n = npg * Nv * max(CS, 1)
        assert n <= self.grid.x * self.grid.y, (n, self.grid)
        cores = [ttnn.CoreCoord(i % self.grid.x, i // self.grid.x) for i in range(n)]
        noc = [self.mesh.worker_core_from_logical_core(c) for c in cores]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        if CS and os.environ.get("QWEN36_GDN_RECT", "1") == "1":
            # v2+: the same cores as row rectangles (dispatch writes binaries / CB configs per range: 96 single-core
            # ranges cost more than a 29-us program at B = 1, CS 4)
            crs = ttnn.num_cores_to_corerangeset(n, self.grid, True)
        # chainall (ep2 t3, DF_GDN_CHAIN1): (bufs, Xl, base) = all T slots of the DFlash2 verify in ONE program
        # (kernels/gdnq4c_*: state base -> bufs[0] -> ... -> bufs[T-1], conv history from the L1 window Xl)
        xconv = chain[4] if chain is not None and len(chain) > 4 else (chainall[1] if chainall is not None else None)
        st = self._conv() if xconv is None else [xconv] * 4
        rec = self._rec() if chain is None and chainall is None else (chain[1] if chain is not None else chainall[2])
        assert (rec.dtype == ttnn.float32) == bool(STF32), (rec.dtype, STF32)
        v3 = CS and self.fast >= 3
        v4 = CS and self.fast >= 4
        # v6: the pair merge only pays with more than one row pair (B = 1: 29.0 vs 28.4 us for the v4 reader);
        # v7 (shared state CB): exact but slower at B = 8 (101 vs 91 us), kept for the record only
        v8 = CS and self.fast >= 8 and npg > 1 and multi is None  # v6 + the first state read by the writer RISC (B = 1: slower)
        v6 = CS and self.fast >= 6 and (npg > 1 or v8) and multi is None  # (multi: npg groups of B = 1, no pair merge)
        v7 = CS and self.fast == 7 and npg > 1 and multi is None
        if group is not None:
            assert v6 and not v7 and not v8 and chain is None and chainall is None and npg > 1, (self.fast, npg)
        if chain is not None or chainall is not None:
            assert B == 1 and v4 and not v6 and not v7 and not v8, "chained verify: B = 1 on the v4 kernels"
        if v3 and self.wpk3 is None:
            raise RuntimeError("QWEN36_GDN_FAST=3 needs the v3 taps (built when the env is set at model build)")
        wk = self.wpk3 if v3 else self.wpk
        if chainall is not None and os.environ.get("DF_GDN_PRO", "0") == "1" and getattr(self, "wbat", None) is not None:
            wk = self.wbat  # GDNQ_PRO: the reader reads whole prebuilt W tiles
        ins = [qkvzab, st[0], st[1], st[2], st[3], wk, self.sel, rec, self.dtbf, self.anegf, self.onw]
        aidx = chainall is not None and len(chainall) > 3 and chainall[3] is not None
        if aidx:  # ep2 t3b DF_NOCOMMIT: the KV-skip position input carries the previous accepted slot
            ins.append(chainall[3])
        rct = [0, self.Nk, Nv, ZT0, ACOL, BCOL, CMODE, STF32] + ([CS] if CS else [])
        for t in ins:
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wct = [Nv, STF32] + ([CS, 1] if CS else []) + ([ACOL, BCOL] if v4 else [])
        for t in (rec, out) + ((qkvzab, self.dtbf, self.anegf) if v4 else ()):
            wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        ts = getattr(self, "ts_tensor", None)  # ep2 gdn: per-core stage timestamps (GDNQ_TS ablation runs only)
        if ts is not None:
            wct.extend(ttnn.TensorAccessorArgs(ts).get_compile_time_args())
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rf = Nv // self.Nk
        for i, c in enumerate(cores):
            if CS:  # v2: core i = (pl * Nv + hv) * CS + ci, pair p = P0 + pl
                pl, hv, ci = i // (Nv * CS), (i // CS) % Nv, i % CS
                p = P0 + pl
                ow = noc[(pl * Nv + (hv - hv % rf)) * CS]
                peers = []
                for cj in range(CS):
                    if cj != ci:
                        pc = noc[(pl * Nv + hv) * CS + cj]
                        peers += [pc.x, pc.y]
                ins_c = ins if multi is None else self._ms_ins(ins, multi[pl])
                rrt[c.x][c.y] = [hv, p, B, ci] + [t.buffer_address() for t in ins_c] + [ow.x, ow.y]
                fw = []
                if v4 and hv % rf == 0 and ci == 0:  # the key-head owner forwards the q / k conv rows to these cores
                    for hj in range(hv, hv + rf):
                        for cj in range(CS):
                            if (hj, cj) != (hv, ci):
                                fc = noc[(pl * Nv + hj) * CS + cj]
                                fw += [fc.x, fc.y]
                if v6:  # + counts; merge (B > 2): the other pairs' cores of this slice (pair 0) / the pair-0 core
                    cons = []
                    if pl == 0 and npg > 1:
                        for pj in range(1, npg):
                            cc = noc[(pj * Nv + hv) * CS + ci]
                            cons += [cc.x, cc.y]
                    sc = noc[hv * CS + ci]
                    fw = [len(fw) // 2] + fw + [len(cons) // 2] + cons + [sc.x, sc.y]
                rrt[c.x][c.y] = list(rrt[c.x][c.y]) + fw
                if chainall is not None:
                    cbufs = chainall[0] if multi is None else multi[pl]["bufs"]
                    rrt[c.x][c.y] = list(rrt[c.x][c.y]) + [b.buffer_address() for b in (cbufs if aidx else cbufs[:-1])]
                    if multi is not None:
                        rrt[c.x][c.y] = list(rrt[c.x][c.y]) + [int(multi[pl]["xrow0"])]
                wx = [qkvzab.buffer_address(), self.dtbf.buffer_address(), self.anegf.buffer_address()] if v4 else []
                last, rec_w = (int(p == npairs - 1), rec) if chain is None else (int(chain[3]), chain[2])
                wrt[c.x][c.y] = [hv, p, B, ci, last, rec_w.buffer_address(), out.buffer_address()] + wx + peers
                if chainall is not None:
                    cbufs = chainall[0] if multi is None else multi[pl]["bufs"]
                    wrt[c.x][c.y] = list(wrt[c.x][c.y]) + [b.buffer_address() for b in cbufs]
                    if multi is not None:
                        m_ = multi[pl]
                        wrt[c.x][c.y] = list(wrt[c.x][c.y]) + [int(m_["xrow0"]), int(m_["orow0"]), int(m_["oz0"])]
                crt[c.x][c.y] = [2 if 2 * p + 1 < B else 1, ci * (4 // CS)]
            else:
                p, hv = i // Nv, i % Nv
                ow = noc[p * Nv + (hv - hv % rf)]
                rrt[c.x][c.y] = [hv, p, B] + [t.buffer_address() for t in ins] + [ow.x, ow.y]
                wrt[c.x][c.y] = [hv, p, B, int(p == npairs - 1), rec.buffer_address(), out.buffer_address()]
                crt[c.x][c.y] = [2 if 2 * p + 1 < B else 1]
            if ts is not None:
                wrt[c.x][c.y] = list(wrt[c.x][c.y]) + [ts.buffer_address(), i]

        kdefs = getattr(self, "kdefs", {})  # ep2 gdn: per-kernel-file defines (variants / ablations), default none

        def kd(src, ct, rt, cfg):
            k = ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/{src}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )
            if chainall is not None:
                k.defines = list(k.defines) + [("GDNQ_CHAIN", str(len(chainall[0]))), ("GDNQ_XCONV", "1")]
                if multi is not None and ("reader" in src or "writer" in src):
                    k.defines = list(k.defines) + [("GDNQ_MSRT", "1")]
                msr = getattr(self, "_ms_rows", None)  # ms1o multi-stream verify: (in rows, out rows, zero-from row)
                if ("reader" in src or "writer" in src) and (
                        ((multi is not None or msr is not None) and os.environ.get("DF_MS_STL1", "1") == "1")
                        or os.environ.get("DF_GDN_STL1", "0") == "1"):  # DF_GDN_STL1: the single-stream verify too
                    k.defines = list(k.defines) + [("GDNQ_STL1", "1"),  # ms1o: the state handed over in L1
                                                   ("GDNQ_STRING", os.environ.get("DF_MS_STRING", "16"))]
                if msr is not None:
                    k.defines = list(k.defines) + [("GDNQ_XROW0", str(int(msr[0]))), ("GDNQ_OROW0", str(int(msr[1]))),
                                                   ("GDNQ_OZ0", str(int(msr[2])))]
                if aidx and "reader" in src:
                    k.defines = list(k.defines) + [("GDNQ_AIDX", "1")]
                if os.environ.get("DF_GATES_AHEAD", "0") == "1" and "writer" in src:
                    k.defines = list(k.defines) + [("GDNQ_GATES_AHEAD", "1")]
                if os.environ.get("DF_GATES_ALL", "0") == "1" and ("writer" in src or "compute" in src):
                    k.defines = list(k.defines) + [("GDNQ_GATES_ALL", "1")]
                if os.environ.get("DF_RMSALL", "0") == "1":
                    k.defines = list(k.defines) + [("GDNQ_RMSALL", "1")]
                if os.environ.get("DF_GDN_PRO", "0") == "1" and ("reader" in src or "compute" in src):
                    k.defines = list(k.defines) + [("GDNQ_PRO", "1")]
                    if os.environ.get("DF_GDN_PRO2", "0") == "1":
                        k.defines = list(k.defines) + [("GDNQ_PRO2", "1")]
                    if os.environ.get("DF_GDN_PRO_ONEBAR", "0") == "1":
                        k.defines = list(k.defines) + [("GDNQ_PRO_ONEBAR", "1")]
            if group is not None and "reader" in src:
                k.defines = list(k.defines) + [("GDNQ_P0", str(P0)), ("GDNQ_NPG", str(npg))]
            if ("reader" in src or "writer" in src) and not src.startswith("gdnq4c"):
                # r64e: qkvzab width in tiles, so the kernels can address the second tile row of a >32-row step
                k.defines = list(k.defines) + [("GDNQ_XT", str(-(-int(qkvzab.shape[-1]) // 32)))]
            if chain is not None and ("reader" in src or "writer" in src):
                k.defines = list(k.defines) + [("GDNQ_XROW", str(int(chain[0])))]
                if xconv is not None and "reader" in src:
                    k.defines = list(k.defines) + [("GDNQ_XCONV", "1")]
            kk = kdefs.get(src) or kdefs.get(src.replace("gdnq2_", "gdnq_").replace("gdnq3_", "gdnq_").replace("gdnq4_", "gdnq_").replace("gdnq5_", "gdnq_").replace("gdnq6_", "gdnq_").replace("gdnq7_", "gdnq_").replace("gdnq8_", "gdnq_"))
            if kk:
                k.defines = list(k.defines) + [(a, str(b)) for a, b in kk.items()]
            return k

        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        state_cbs = (8, 13, 17)  # the recurrent state in (SFPU copy), in (matmul copy), out: the state's own format
        utd = [ttnn.UnpackToDestMode.Default] * 64
        cbs = []
        cbmap = dict(CBS)
        string = int(os.environ.get("DF_MS_STRING", "16"))  # ms1o: cb_state ring (tiles) of the multi-stream chains
        if chainall is not None and (multi is not None or getattr(self, "_ms_rows", None) is not None) and string != 16:
            cbmap[17] = (string, 1, 0)
        if ts is not None:
            cbmap[27] = (1, 0, 0)
        if CS:  # v2: o double-buffered (peers write the next sequence's tiles early), osig / orecv handshakes
            cbmap[28] = (8, 1, 0)
            cbmap[32] = (1, 0, 0)
            cbmap[33] = (1, 0, 0)
        if v3:
            cbmap[11] = (24, 0, 0)  # selectors of both sequences
        if CS and self.fast == 5:
            cbmap[20] = (2, 1, 0)  # the q and k norm scalars
            cbmap[29] = (2, 1, 0)
        if v6:
            cbmap[34] = (12, 0, 0)  # merge staging: 4 tensors x 12 channel tiles x 2 faces x 256 bytes
        if v4:
            cbmap[30] = (2, 1, 1)  # decay / beta of both sequences (gates first)
            cbmap[31] = (2, 1, 1)
        gates_ahead = chainall is not None and os.environ.get("DF_GATES_AHEAD", "0") == "1"
        if chainall is not None and os.environ.get("DF_GDN_PRO", "0") == "1":  # ep2 t3b GDNQ_PRO prologue CBs
            nt9 = 8 + max(1, 4 // max(CS, 1))
            onebar = os.environ.get("DF_GDN_PRO_ONEBAR", "0") == "1"
            cbmap[37] = (4 * nt9 if onebar else 8, 0, 0)  # A: conv input rows of all slots, 4 tiles per channel tile
            cbmap[38] = (4 * nt9 if onebar else 8, 0, 0)  # W: taps on every row
            cbmap[39] = (1, 1, 0)        # rsqrt'd gram
            cbmap[40] = (1, 1, 0)        # its diagonal as column 0
            cbmap[41] = (nt9, 0, 0)      # conv output, rows = slots
            cbmap[42] = (4, 1, 0)        # q norm'd, rows = slots
            cbmap[43] = (4, 1, 0)        # k norm'd, rows = slots
            if os.environ.get("DF_GDN_PRO2", "0") == "1":
                cbmap[44] = (4, 1, 0)    # k as columns (slots)
                cbmap[45] = (1, 1, 0)    # q . k of all slots (diagonal)
        if chainall is not None and os.environ.get("DF_RMSALL", "0") == "1":  # ep2 t3b: rms / out of all slots at the end
            cbmap[46] = (4, 1, 0)  # o_all: rows = slots, all 4 column tiles
            cbmap.setdefault(39, (1, 1, 0))
            cbmap.setdefault(40, (1, 1, 0))
        gates_all = chainall is not None and os.environ.get("DF_GATES_ALL", "0") == "1"
        if gates_all:  # ep2 t3b: the gates of all T slots in one compute pass (cb 35 / 36 = decay / beta rows)
            cbmap[35] = (1, 1, 0)
            cbmap[36] = (1, 1, 0)
        if gates_ahead:  # ep2 t3b: the writer fills the gate inputs of all T slots first
            cbmap[4] = (len(chainall[0]), 0, 0)
            cbmap[5] = (len(chainall[0]), 0, 0)
        if v7:  # cb_smm (13) and cb_sdst (8) share one allocation of 32 state tiles (both sequences of the pair)
            cbmap.pop(8)
            cbmap[13] = (32, 1, 0)
        for i, (nt, f32, u) in cbmap.items():
            if i in state_cbs:
                f32 = STF32
                u = u and STF32
            if u:
                utd[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ps = FP32_TILE if f32 else BF16_TILE
            fmt = ttnn.float32 if f32 else ttnn.bfloat16
            fds = [ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=ps)]
            if v7 and i == 13:
                fds.append(ttnn.CBFormatDescriptor(buffer_index=8, data_format=fmt, page_size=ps))
                if STF32:
                    utd[8] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cbs.append(ttnn.CBDescriptor(total_size=ps * nt, core_ranges=crs, format_descriptors=fds))
        cc.unpack_to_dest_mode = utd
        pre = "gdnq4" if v4 else "gdnq3" if v3 else "gdnq2" if CS else "gdnq"
        kernels = [
            kd(f"{'gdnq8' if v8 else 'gdnq7' if v7 else 'gdnq6' if v6 else pre}_reader.cpp", rct, rrt,
               ttnn.ReaderConfigDescriptor()),
            kd(f"{'gdnq8' if v8 else 'gdnq4' if v4 else 'gdnq2' if CS else 'gdnq'}_writer.cpp", wct, wrt,
               ttnn.WriterConfigDescriptor()),
            kd(f"{'gdnq5' if CS and self.fast == 5 else pre}_compute.cpp", self.ct + ([CS] if CS else []), crt, cc),
        ] if chainall is None else [
            kd("gdnq4c_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("gdnq4c_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("gdnq4c_compute.cpp", self.ct + [CS], crt, cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)]
        if CS:
            sems.append(ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0))
        if v6:
            sems.append(ttnn.SemaphoreDescriptor(id=2, core_ranges=crs, initial_value=0))
        if chainall is not None:  # writer -> reader on the same core: states stored so far
            sems.append(ttnn.SemaphoreDescriptor(id=3, core_ranges=crs, initial_value=0))
        extra = [chain[2]] if chain is not None else (list(chainall[0]) if chainall is not None else [])
        if multi is not None:
            extra = []
            for m_ in multi:
                extra += [m_["Xl"], m_["base"], m_["kvm"]] + list(m_["bufs"])
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), ins + ([ts] if ts is not None else []) + extra + [out]

    def __call__(self, qkvzab, B):
        """qkvzab: the in-projection output [1, B, W] (bf16 TILE, L1 or DRAM interleaved) -> gated [1, B, Nv*128]
        (L1). Updates the conv state and the recurrent state rows [0:B] in place."""
        assert qkvzab.dtype == ttnn.bfloat16 and qkvzab.layout == ttnn.TILE_LAYOUT, (qkvzab.dtype, qkvzab.layout)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, B, self.Nv * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG
        )
        # one ProgramDescriptor = the same program on every device of the mesh (SPMD); the per-device shards of every
        # tensor sit at the same addresses
        g = self.group_pairs(B)
        if g is None:
            prog, io = self._program(qkvzab, B, out)
            return ttnn.generic_op(io, prog)
        # t3c (B > 8): one program per group of g row pairs, in order (each updates its own state rows / output rows)
        npairs = (B + 1) // 2
        for P0 in range(0, npairs, g):
            prog, io = self._program(qkvzab, B, out, group=(P0, min(g, npairs - P0)))
            ttnn.generic_op(io, prog)
        return out

    def group_pairs(self, B):
        """t3c: None when all (B + 1) / 2 row pairs fit the grid in one program (B <= 8 here: unchanged), else the
        pairs per program (QWEN36_GDN_GROUP caps it; default the most that fit at one core per (head, pair))."""
        npairs = (B + 1) // 2
        cap = self.grid.x * self.grid.y
        if npairs * self.Nv <= cap:
            return None
        g = int(os.environ.get("QWEN36_GDN_GROUP", "0")) or cap // self.Nv
        return max(2, min(g, cap // self.Nv))


    def chain_call(self, qkvzab, out, xr, rec_in, rec_out, last, xconv=None):
        """ep2 t3 (DFlash2 verify): one chained slot = the B = 1 program on row xr of qkvzab [1, T, W], state rec_in ->
        rec_out (row 0 each), gated row -> row xr of out [1, T, Nv*128] (L1). Same math as a B = 1 call on that row."""
        chain = (xr, rec_in, rec_out, last, xconv)
        if os.environ.get("DF_PCACHE", "1") != "1":
            prog, io = self._program(qkvzab, 1, out, chain=chain)
            return ttnn.generic_op(io, prog)
        # the descriptor depends only on the tensors' addresses (+ xr / last): rebuild only when one moved (a re-capture
        # after a prefill usually lands every tensor where it was; building one costs ~1 ms of Python per slot)
        conv = self._conv() if xconv is None else [xconv] * 4
        key = (xr, bool(last)) + tuple(t.buffer_address() for t in [qkvzab, out, rec_in, rec_out] + list(conv))
        pc = self.__dict__.setdefault("_pcache", {})
        hit = pc.get(xr)
        if hit is None or hit[0] != key:
            assert getattr(self, "ts_tensor", None) is None
            hit = (key,) + self._program(qkvzab, 1, out, chain=chain)
            pc[xr] = hit
        # same descriptor, the current tensor objects (io order of _program: qkvzab, st0..st3, taps, sel, rec, dt_bias,
        # A_neg, norm w, rec_out, out)
        io = list(hit[2])
        io[0], io[1:5], io[7], io[-2], io[-1] = qkvzab, list(conv), rec_in, rec_out, out
        return ttnn.generic_op(io, hit[1])


    @staticmethod
    def _ms_ins(ins, m_):
        """ins of _program with stream m_'s window (st0..st3), base state (rec) and `a` input (kvm, last)."""
        r = list(ins)
        r[1:5] = [m_["Xl"]] * 4
        r[7] = m_["base"]
        r[11] = m_["kvm"]
        return r

    def chain_multi(self, qkvzab, out, multi):
        """ms1o: the chained verify slots of several streams in ONE program (stream s on its own core group, see
        _program multi). multi[s] = dict(Xl, base, bufs, kvm, xrow0, orow0, oz0). The same per-core computation as
        chain_all on each stream (CS may be smaller: per tile the same instruction sequence)."""
        m0 = multi[0]
        ca = (list(m0["bufs"]), m0["Xl"], m0["base"], m0["kvm"])
        key = tuple(t.buffer_address() for t in [qkvzab, out])
        for m_ in multi:
            key = key + tuple(t.buffer_address() for t in [m_["Xl"], m_["base"], m_["kvm"]] + list(m_["bufs"]))
            key = key + (int(m_["xrow0"]), int(m_["orow0"]), int(m_["oz0"]))
        key = key + tuple(os.environ.get(k, "0") for k in ("DF_GATES_AHEAD", "DF_GATES_ALL", "DF_GDN_PRO", "DF_GDN_PRO_ONEBAR", "DF_GDN_PRO2", "DF_RMSALL", "QWEN36_GDN_CS", "DF_MS_STL1", "DF_MS_STRING"))
        pc = self.__dict__.setdefault("_pcache_multi", {})
        ck = (len(multi), int(m0["xrow0"]))  # one entry per stream group (DF_MS_MULTI_G)
        hit = pc.get(ck)
        if hit is None or hit[0] != key:
            prog, io = self._program(qkvzab, 1, out, chainall=ca, multi=multi)
            hit = (key, prog, len(io))
            pc[ck] = hit
        # the io tensors from this call (a cached program may have been built with other tensor objects that sat at
        # the same addresses): ins of stream 0 (12 entries with the `a` input), then every stream's own tensors, out
        ins = self._ms_ins([qkvzab, None, None, None, None, self.wbat if (os.environ.get("DF_GDN_PRO", "0") == "1"
                                                                        and getattr(self, "wbat", None) is not None)
                            else (self.wpk3 if self.fast >= 3 else self.wpk), self.sel, None, self.dtbf, self.anegf,
                            self.onw, None], m0)
        io = ins
        for m_ in multi:
            io = io + [m_["Xl"], m_["base"], m_["kvm"]] + list(m_["bufs"])
        io = io + [out]
        assert len(io) == hit[2], (len(io), hit[2])
        return ttnn.generic_op(io, hit[1])

    def chain_all(self, qkvzab, out, base, bufs, Xl, kvm=None):
        """ep2 t3 (DF_GDN_CHAIN1): all len(bufs) chained verify slots in ONE program (= len(bufs) chain_call()s with
        XCONV, same math per slot): state base -> bufs[0] -> ... -> bufs[T-1], gated rows 0..T-1 into out.
        t3b kvm (DF_NOCOMMIT): slot 0 starts from bufs[a] when the input's element T*T holds a < T."""
        ca = (list(bufs), Xl, base, kvm)
        key = tuple(t.buffer_address() for t in [qkvzab, out, base, Xl] + list(bufs) + ([kvm] if kvm is not None else []))
        key = key + tuple(os.environ.get(k, "0") for k in ("DF_GATES_AHEAD", "DF_GATES_ALL", "DF_GDN_PRO", "DF_GDN_PRO_ONEBAR", "DF_GDN_PRO2", "DF_RMSALL"))
        key = key + (getattr(self, "_ms_rows", None), os.environ.get("DF_MS_STL1", "1"), os.environ.get("DF_MS_STRING", "16"),
                     os.environ.get("DF_GDN_STL1", "0"))
        pc = self.__dict__.setdefault("_pcache", {})
        hit = pc.get("all")
        if hit is None or hit[0] != key or os.environ.get("DF_PCACHE", "1") != "1":
            hit = (key,) + self._program(qkvzab, 1, out, chainall=ca)
            pc["all"] = hit
        io = list(hit[2])
        T = len(bufs)
        io[0], io[1:5], io[7] = qkvzab, [Xl] * 4, base
        io[-1 - T:-1] = list(bufs)
        io[-1] = out
        if kvm is not None:
            io[11] = kvm
        return ttnn.generic_op(io, hit[1])


def _per_device(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


class FusedGDNDecodeTP(_FusedGDNBase):
    """TPGatedDeltaNet (the 2-card path): qkvzab [q | k | v | z | a | b] per device, conv_states[0..3] (4-deep shift
    register), fp32 (bf16 with QWEN35_GDN_STATE_BF16=1) rec_state [Bmax, Nv, 128, 128]."""

    def __init__(self, gdn):
        self.gdn = gdn
        Nk, Nv = gdn.Nk, gdn.Nv
        assert gdn.Dk == 128 and gdn.Dv == 128 and Nv % Nk == 0 and gdn.K == 4, (Nk, Nv, gdn.Dk, gdn.Dv, gdn.K)
        self.mesh, self.Nk, self.Nv = gdn.mesh, Nk, Nv
        tw = gdn.tw
        st_f32 = int(os.environ.get("QWEN35_GDN_STATE_BF16") != "1")
        ab = (8 * Nk + 8 * Nv) * 32
        self.layout = (8 * Nk + 4 * Nv, ab, ab + Nv, 1, st_f32)
        taps = [_per_device(t) for t in tw["conv_taps"]]
        self._build(
            taps,
            _per_device(tw["dt_bias"]),
            _per_device(tw["neg_exp_A"]),
            _per_device(tw["norm_w"])[0],
            gdn.qkv_dim_tp,
            1e-6,
        )

    def _conv(self):
        return list(self.gdn.conv_states)

    def _rec(self):
        return self.gdn.rec_state


class FusedGDNDecode1(_FusedGDNBase):
    """Qwen36GatedDeltaNet (the 1-card path, B = 1): mega [q | k | v | a | b | z] (one matmul), split_conv_state =
    3 history rows [1, 1, D], bf16 recurrent_state [1, Nv, 128, 128]."""

    def __init__(self, dn):
        self.dn = dn
        w = dn.weights
        Nk, Nv = dn.num_heads, dn.num_v_heads
        assert dn.head_k_dim == 128 and dn.head_v_dim == 128 and dn.conv_kernel_size == 4, dn.head_k_dim
        assert w.mega_fused_weight is not None and w.fused_conv_bias_dev is None
        qkv = w.mega_qkv_dim
        assert qkv == 2 * Nk * 128 + Nv * 128 and (qkv + 2 * Nv) % 32 == 0, qkv
        self.mesh, self.Nk, self.Nv = dn.device, Nk, Nv
        self._layout0 = ((qkv + 2 * Nv) // 32, qkv, qkv + Nv)
        taps = [[ttnn.to_torch(t).float()] for t in w.fused_conv_weight_taps]
        self._build(
            taps,
            [ttnn.to_torch(w.dt_bias).float()],
            [ttnn.to_torch(w.A_neg).float()],
            ttnn.to_torch(w.o_norm_weight).float(),
            qkv,
            dn.norm_eps,
        )

    @property
    def layout(self):
        # the vLLM path keeps the fused [1, 3, D] conv state (split_conv_state is None); the split rows if they exist
        return self._layout0 + ((0 if self.dn.split_conv_state is not None else 2), 0)

    def ready(self):
        return self.dn.recurrent_state is not None and (
            self.dn.split_conv_state is not None or self.dn.fused_conv_state is not None
        )

    def _conv(self):
        if self.dn.split_conv_state is not None:
            s = list(self.dn.split_conv_state)
            return [s[0], s[0], s[1], s[2]]
        f = self.dn.fused_conv_state
        assert list(f.shape)[:2] == [1, 3] and f.layout == ttnn.TILE_LAYOUT and f.dtype == ttnn.bfloat16, f.shape
        return [f, f, f, f]

    def _rec(self):
        return self.dn.recurrent_state

    def forward(self, x, ckc):
        """The whole one-token mixer for x [1, 1, H]: mega in-projection, fused body, o_proj (as the chain: L1)."""
        w = self.dn.weights
        L1 = ttnn.L1_MEMORY_CONFIG
        mega = ttnn.linear(x, w.mega_fused_weight, memory_config=L1, compute_kernel_config=ckc)
        o = self(mega, 1)
        ttnn.deallocate(mega)
        y = ttnn.linear(o, w.o_proj_weight, memory_config=L1, compute_kernel_config=ckc)
        ttnn.deallocate(o)
        return y
