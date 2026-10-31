# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused one-token GDN decode (port5 Track B, opt-in: DRAMA3_GDN_FUSED_DECODE=all or a comma list of layer indices).

The decode GDN mixer is ~56 programs (tests/profile_ops_dev.py: 369 us per layer, of which the two weight matmuls,
the mega in-projection and o_proj, are ~97 us). Everything between those two matmuls (the slices of
[qkv | a | b | z], the partial-sum conv, the q / k l2 norms, the gates, the delta-rule recurrence, the gated output
norm) runs here as ONE program (kernels/gdn_decode_*.cpp via ttnn.generic_op): 16 cores, one per key head (= two
value heads, so no conv channel is shared between cores), state and conv partial sums updated in place.

Contract (Drama 3 decode with the --fast defaults): mega-fused in-projection, ConvPartialSums conv state
(DRAMA3_GDN_CONV_PARTIALS=1), no conv bias, external bf16 recurrent state [1, 32, 128, 128], 16 key / 32 value
heads of 128. The math follows the lean QSCALE decode path; it is not bit-identical (fp32 DST through the chain,
bf16 only at the CB hand-offs), so it sits behind the golden check like any other change.
"""

import os
import struct

import torch

import ttnn

KDIR = "models/demos/blackhole/drama3/kernels"
TILE_BYTES = 2048  # bf16 tile
# CB index -> tiles (bf16). Inputs 0..10 (reader), outputs 16..18 (writer), the rest compute-internal.
CBS = {
    0: 2, 1: 6, 2: 8, 3: 8, 4: 1, 5: 1, 6: 1, 7: 1, 8: 16, 9: 4, 10: 2,
    16: 4, 17: 16, 18: 6, 19: 16, 20: 1, 21: 4, 22: 4, 23: 4, 24: 16, 25: 4, 26: 4, 27: 16, 28: 4, 29: 4, 30: 1, 31: 1,
}
# v2 (DRAMA3_GDN_FUSED_V=2, packed conv): x / Q / w / Q' as one packed tile each, + selectors, staging, packed y
CBS2 = {**CBS, 0: 1, 1: 3, 2: 4, 18: 3, 11: 16, 12: 4, 13: 3, 15: 1}
# v3 (DRAMA3_GDN_FUSED_V=3): v2 with one core per value head (32 cores, 12 packed channel rows, one head per core)
CBS3 = {**CBS2, 3: 4, 11: 12, 19: 12}


def channel_tiles(hk):
    """The 16 conv channel tiles of key head hk: q 4hk.., k 64+4hk.., v 128+8hk.. (value heads 2hk, 2hk+1)."""
    return [4 * hk + t for t in range(4)] + [64 + 4 * hk + t for t in range(4)] + [128 + 8 * hk + t for t in range(8)]


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def layers_from_env(n_layers):
    v = os.environ.get("DRAMA3_GDN_FUSED_DECODE", "")
    if not v:
        return set()
    return set(range(n_layers)) if v == "all" else {int(i) for i in v.split(",")}


class FusedGDNDecode:
    def __init__(self, dn, mesh):
        w = dn.weights
        assert w.mega_fused_weight is not None and w.fused_conv_bias_dev is None
        assert getattr(dn.split_conv_state, "partial_sums", False), "needs DRAMA3_GDN_CONV_PARTIALS=1"
        H, Hk, K, V = dn.num_v_heads, dn.num_heads, dn.head_k_dim, dn.head_v_dim
        assert (H, Hk, K, V) == (32, 16, 128, 128) and w.mega_qkv_dim == 8192, (H, Hk, K, V, w.mega_qkv_dim)
        self.dn, self.mesh = dn, mesh
        up = lambda t: ttnn.from_torch(  # noqa: E731
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def per_head(vals):  # head h's value at [0, 0] of tile h
            t = torch.zeros(H * 32, 32)
            t[torch.arange(H) * 32, 0] = vals.float()
            return up(t)

        self.aneg = per_head(ttnn.to_torch(w.A_neg).float().flatten()[:H])
        self.dtb = per_head(ttnn.to_torch(w.dt_bias).float().flatten()[:H])
        self.onw = up(ttnn.to_torch(w.o_norm_weight).float().flatten()[:V].reshape(1, 1, V))
        self.out = up(torch.zeros(1, 1, H * V))  # persistent output (allocated before any trace capture)
        # v2: the conv taps packed per core (tile 4 hk + tap: row t = channel tile t's 32 taps), row selectors E_t
        self.ver = int(os.environ.get("DRAMA3_GDN_FUSED_V", "1"))
        self.v2 = self.ver >= 2
        if self.v2:
            taps = [ttnn.to_torch(tp).float().flatten()[:8192].reshape(256, 32) for tp in w.fused_conv_weight_taps]
            n_cores = H if self.ver == 3 else Hk
            wp = torch.zeros(n_cores * 4 * 32, 32)
            for p in range(n_cores):
                ch = channel_tiles(p) if self.ver == 2 else channel_tiles(p // 2)[:8] + [128 + 4 * p + t for t in range(4)]
                for tap in range(4):
                    wp[(p * 4 + tap) * 32 : (p * 4 + tap) * 32 + len(ch)] = taps[tap][ch]
            self.wpk = up(wp)
            sel = torch.zeros(16 * 32, 32)
            sel[torch.arange(16) * 32, torch.arange(16)] = 1.0
            self.sel = up(sel)
        self.ct = [_bits(K**-0.5), _bits(1e-6), _bits(1.0), _bits(1.0 / V), _bits(dn.norm_eps), _bits(20.0)]
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(H if self.ver == 3 else Hk)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.noc = [mesh.worker_core_from_logical_core(c) for c in self.cores]
        self._desc = None
        self.out_b = {}  # port6: the [1, B, 4096] L1 outputs of forward_batch

    def _program(self, mega):
        dn, w = self.dn, self.dn.weights
        Q = list(dn.split_conv_state)
        taps = list(w.fused_conv_weight_taps)
        st = dn.recurrent_state
        if self.v2:
            ins = [mega, Q[0], Q[1], Q[2], self.wpk, self.sel, st, self.dtb, self.aneg, self.onw]
        else:
            ins = [mega, Q[0], Q[1], Q[2], taps[0], taps[1], taps[2], taps[3], st, self.dtb, self.aneg, self.onw]
        v = {1: "gdn_decode", 2: "gdn_decode2", 3: "gdn_decode3"}[self.ver]
        rct = [0] if self.ver == 3 else []  # v3: the semaphore id
        for t in ins:
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wouts = [Q[0], Q[1], Q[2], st, self.out]
        wct = [0] if self.ver == 3 else []
        for t in wouts:
            wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for p, c in enumerate(self.cores):
            ra = [p] + [t.buffer_address() for t in ins]
            if self.ver == 3:  # the NOC coordinates of the even core of this key head (the odd core signals it)
                ra += [self.noc[p & ~1].x, self.noc[p & ~1].y]
            rrt[c.x][c.y] = ra
            wrt[c.x][c.y] = [p] + [t.buffer_address() for t in wouts]
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{v}_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=rct,
            runtime_args=rrt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{v}_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=wct,
            runtime_args=wrt,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{v}_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=self.ct,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=TILE_BYTES * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=TILE_BYTES)],
            )
            for i, n in {1: CBS, 2: CBS2, 3: CBS3}[self.ver].items()
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.crs, initial_value=0)] if self.ver == 3 else []
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs), ins + [self.out]

    def __call__(self, mega):
        """mega: the in-projection output [1, 1, 12352] (bf16 TILE) -> o [1, 1, 4096] (this object's persistent buffer;
        the recurrent state and conv partial sums are updated in place)."""
        prog, io = self._program(mega)
        return ttnn.generic_op(io, prog)

    # ---------------------------------------------------------------- port6: B sequences (v4 kernels)
    def _program_batch(self, mega, B):
        """v4: core (hv, g) runs value head hv of the sequences g, g + G, .. < B (G = min(B, 3): up to 96 cores).
        Layouts: mega [1, B, 12352]; dn.split_conv_state = the batch partial sums [1, 2B, 8192] (sequence b in row
        2b); dn.recurrent_state [B, 32, 128, 128]; output [1, B, 4096] in L1."""
        # port10: up to B = 16 the 2B partial-sum rows fill one tile (both face halves); 16 < B <= 32: v5 kernels (two tile rows)
        assert self.ver == 3 and B <= 32
        kv = "gdn_decode4" if B <= 16 else "gdn_decode5"
        dn = self.dn
        Q = list(dn.split_conv_state)
        assert Q[0].shape[-2] == 2 * B, ("batch partial sums need the [1, 2B, D] layout", Q[0].shape)
        st = dn.recurrent_state
        assert st.shape[0] == B
        # a fresh L1 tensor per call (a persistent L1 buffer clashed with the static CBs of the card codec); 32-byte
        # row writes need L1 (16-byte alignment), DRAM would need 64
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        self.out_b[B] = out  # the last output (tests read it)
        G = min(B, 3)
        g = self.mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(32 * G)]
        noc = [self.mesh.worker_core_from_logical_core(c) for c in cores]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        ins = [mega, Q[0], Q[1], Q[2], self.wpk, self.sel, st, self.dtb, self.aneg, self.onw]
        rct = [0]
        for t in ins:
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wouts = [Q[0], Q[1], Q[2], st, out]
        wct = [0]
        for t in wouts:
            wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for p, c in enumerate(cores):
            hv, gi = p % 32, p // 32
            even = noc[gi * 32 + (hv & ~1)]
            rrt[c.x][c.y] = [hv, gi, G, B] + [t.buffer_address() for t in ins] + [even.x, even.y]
            wrt[c.x][c.y] = [hv, gi, G, B] + [t.buffer_address() for t in wouts]
            crt[c.x][c.y] = [len(range(gi, B, G))]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd(f"{kv}_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd(f"{kv}_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("gdn_decode4_compute.cpp", self.ct, crt, ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=TILE_BYTES * n,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=TILE_BYTES)],
            )
            for i, n in CBS3.items()
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), ins + [out]

    def forward_batch(self, x, ckc):
        """The decode mixer for B one-token sequences x [1, B, H] (rows_are_batch): mega in-projection, the v4
        fused body, o_proj -> [1, B, H]."""
        w = self.dn.weights
        L1 = ttnn.L1_MEMORY_CONFIG
        B = x.shape[-2]
        mega = ttnn.linear(x, w.mega_fused_weight, memory_config=L1, compute_kernel_config=ckc)
        prog, io = self._program_batch(mega, B)
        o = ttnn.generic_op(io, prog)
        ttnn.deallocate(mega)
        y = ttnn.linear(o, w.o_proj_weight, memory_config=L1, compute_kernel_config=ckc)
        ttnn.deallocate(o)
        return y

    def forward(self, x, ckc):
        """The whole decode mixer for a normed one-token row x [1, 1, H]: mega in-projection, fused body, o_proj."""
        w = self.dn.weights
        L1 = ttnn.L1_MEMORY_CONFIG
        mega = ttnn.linear(x, w.mega_fused_weight, memory_config=L1, compute_kernel_config=ckc)
        o = self(mega)
        ttnn.deallocate(mega)
        return ttnn.linear(o, w.o_proj_weight, memory_config=L1, compute_kernel_config=ckc)


def batch_on():
    """DRAMA3_GDN_FUSED_BATCH=1 (port6, opt-in): batch-mode GDN decode through the v4 kernels (needs the batch
    partial sums in the [1, 2B, D] layout; host/batch.py allocates them so when this is on)."""
    return os.environ.get("DRAMA3_GDN_FUSED_BATCH", "0") == "1"


def install(layers, mesh, indices):
    """Route the one-token decode of the GDN layers in indices through FusedGDNDecode (prefill unchanged)."""
    n = 0
    for i, layer in enumerate(layers):
        if i not in indices or layer.is_full_attention:
            continue
        dn = layer.attention
        fused = FusedGDNDecode(dn, mesh)
        orig = dn.forward

        def fwd(x, mode="recurrent", chunk_size=None, valid_len=None, _orig=orig, _f=fused, _dn=dn):
            if mode == "recurrent" and x.shape[-2] == 1 and not getattr(_dn, "rows_are_batch", False):
                return _f.forward(x, _dn.compute_kernel_config_decode)
            if mode == "recurrent" and getattr(_dn, "rows_are_batch", False) and batch_on():
                return _f.forward_batch(x, _dn.compute_kernel_config_decode)
            return _orig(x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

        dn.forward = fwd
        dn.fused_decode = fused
        n += 1
    return n
