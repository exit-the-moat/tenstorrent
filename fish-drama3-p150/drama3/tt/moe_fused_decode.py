# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused pieces of the one-token MoE decode (port5 Track B, opt-in).

DRAMA3_MOE_FUSED_TOPK=1: the router's top-k (FillPad + TopK + the index untilize, ~39 us of device time per layer,
tests/profile_ops_dev.py) as one data-movement program on one core (kernels/router_topk.cpp): the top-8 scores of
the 256 router logits and their expert ids, ready for sparse_matmul (uint16 ROW_MAJOR). The softmax over the 8
(QWEN36_ROUTER_TOPK_FIRST) stays a ttnn op.
"""

import os

import torch

import ttnn

KDIR = "models/demos/blackhole/drama3/kernels"


class RouterTopK:
    def __init__(self, mesh, E=256, k=8, core=(0, 0)):
        assert E % 32 == 0 and k <= 16
        self.mesh, self.E, self.k = mesh, E, k
        c = ttnn.CoreCoord(*core)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
        self.core = c
        # port8 DRAMA3_MOE_TOPK_SOFTMAX=1: the softmax over the k scores in the same program (kernels/router_topk_sm.cpp
        # + router_softmax_compute.cpp); __call__ then returns the normalised weights
        self.softmax = os.environ.get("DRAMA3_MOE_TOPK_SOFTMAX", "0") == "1"
        if self.softmax:
            self.ones = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    def __call__(self, scores):
        """scores [1, 1, 1, E] bf16 TILE -> (top-k scores [1, 1, 1, k] bf16 TILE, ids [1, 1, 1, k] uint16 ROW_MAJOR);
        fresh output tensors each call (the MoE block frees them)."""
        L1 = ttnn.L1_MEMORY_CONFIG
        val = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)
        idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.k]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, L1)
        ct = [self.E, self.k]
        for t in (scores, val, idx) + ((self.ones,) if self.softmax else ()):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address()]
        if self.softmax:
            rt[self.core.x][self.core.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address(), self.ones.buffer_address()]
        kern = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/router_topk_sm.cpp" if self.softmax else f"{KDIR}/router_topk.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        nt = self.E // 32
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * (nt + 1),
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            ),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=64)],
            ),
        ]
        kernels = [kern]
        ios = [scores, val, idx]
        if self.softmax:
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/router_softmax_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs, compile_time_args=[], runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
            ))
            cbs += [
                ttnn.CBDescriptor(total_size=2048, core_ranges=self.crs,
                                  format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
                for i in (2, 3, 4, 5, 6, 16)
            ]
            ios = [scores, self.ones, val, idx]
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return val, idx


def install_topk(layers, mesh):
    """Route every MoE router's one-token topk() through RouterTopK (needs QWEN36_ROUTER_TOPK_FIRST=1)."""
    if os.environ.get("DRAMA3_MOE_FUSED_TOPK", "0") != "1":
        return 0
    assert os.environ.get("QWEN36_ROUTER_TOPK_FIRST", "0") == "1"
    rk = RouterTopK(mesh)
    n = 0
    for layer in layers:
        r = getattr(layer.feed_forward, "router", None)
        if r is None or not r.norm_topk_prob:
            continue
        orig = r.topk

        def topk(hidden_states, _r=r, _orig=orig):
            if hidden_states.shape[-2] != 1:
                return _orig(hidden_states)
            s = ttnn.linear(hidden_states, _r.proj_weight, compute_kernel_config=_r.compute_kernel_config)
            raw, idx = rk(s)
            s.deallocate(True)
            if rk.softmax:
                return raw, idx
            v = ttnn.softmax(raw, dim=-1)
            raw.deallocate(True)
            return v, idx

        r.topk = topk
        n += 1
    return n


class MoECombine:
    """out = sum_j v_j * down_j + sigmoid(g) * shared for one token (kernels/moe_combine_*.cpp), the hidden tiles
    spread over n_cores cores. down: the compact [1, k, 32, H] sparse_matmul output (row 0 of each slot),
    v: the softmax'd routing weights [1, 1, 1, k], shared: the shared expert MLP output [1, 1, H], g: the shared
    gate pre-activation (one value at [0, 0])."""

    def __init__(self, mesh, H=2048, k=8, n_cores=None):
        n_cores = n_cores or int(os.environ.get("DRAMA3_MOE_COMBINE_CORES", "16"))
        self.mesh, self.H, self.k = mesh, H, k
        self.HT = H // 32
        assert self.HT % n_cores == 0
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.per = self.HT // n_cores
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, down, v, shared, g, shared_in_down=False):
        """shared_in_down (port10): shared is None, the shared expert's rows are slot k of down."""
        L1 = ttnn.L1_MEMORY_CONFIG
        shared = down if shared_in_down else shared
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)
        rct = [self.k, self.HT]
        for t in (down, v, shared, g):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            t0 = i * self.per
            rrt[c.x][c.y] = [t0, self.per, down.buffer_address(), v.buffer_address(), shared.buffer_address(), g.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kd("moe_combine_s_reader.cpp" if shared_in_down else "moe_combine_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("moe_combine_compute.cpp", [self.k, self.per], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: 1, 1: self.k, 2: 2 * self.k, 3: 2, 4: 1, 5: 1, 16: 2}
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * n,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in sizes.items()
        ]
        ttnn.generic_op([down, v, g, out] if shared_in_down else [down, v, shared, g, out],
                        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


_EXPERT_MM = {}


def shared_in_emm():
    """DRAMA3_MOE_SHARED_IN_EMM=1 (port10, opt-in, with DRAMA3_MOE_EXPERT_MM=1 V=2 and the fused combine / SwiGLU):
    the shared expert's [w3 | w1] and w2 stream as a 9th slot of the routed ExpertMM programs (one program per
    matmul for routed + shared, no separate shared-expert matmuls), MoECombine reads its rows from slot k."""
    return (os.environ.get("DRAMA3_MOE_SHARED_IN_EMM", "0") == "1" and os.environ.get("DRAMA3_MOE_EXPERT_MM", "0") == "1"
            and os.environ.get("DRAMA3_MOE_EXPERT_MM_V", "1") == "2")


def _down_shared(x4, idx, weights, config, sf):
    """port10: routed + shared expert through ExpertMM (expert_mm5): [1, k + 1, 1, H], slot k = the shared expert."""
    mesh = x4.device()
    if "gu5" not in _EXPERT_MM:
        kb = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_KB", "16"))
        _EXPERT_MM["gu5"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_CORES_GU", "64")), kb)
        _EXPERT_MM["dn5"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_CORES_DN", "64")), kb)
        _EXPERT_MM["sw5"] = Swiglu(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_SWIGLU_CORES", "48")))
    k = config.top_k
    w = weights
    ws_gu = ttnn.reshape(sf.w_gu, (1, 1, sf.w_gu.shape[-2], sf.w_gu.shape[-1]))
    ws_dn = ttnn.reshape(sf.w2, (1, 1, sf.w2.shape[-2], sf.w2.shape[-1]))
    up_gate = _EXPERT_MM["gu5"](x4, w.gate_up_proj, idx, k + 1, True, ws=ws_gu)
    act = _EXPERT_MM["sw5"](up_gate)
    up_gate.deallocate(True)
    down = _EXPERT_MM["dn5"](act, w.down_proj, idx, k + 1, False, ws=ws_dn)
    act.deallocate(True)
    return down


def _down_indexed(hidden_states, top_k_indices, weights, config, sparsity, swiglu=None):
    """decode_forward_indexed (qwen36 moe/decode.py) up to the compact down output [1, k, 32, H]; with
    DRAMA3_MOE_EXPERT_MM=1 (needs the fused top-k ids and the SwiGLU kernel) both expert matmuls run as ExpertMM."""
    if os.environ.get("DRAMA3_MOE_EXPERT_MM", "0") == "1" and swiglu is not None and top_k_indices.layout == ttnn.ROW_MAJOR_LAYOUT:
        mesh = hidden_states.device()
        if not _EXPERT_MM:
            kb = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_KB", "16"))
            _EXPERT_MM["gu"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_EXPERT_MM_CORES_GU", "64")), kb)
            _EXPERT_MM["dn"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_EXPERT_MM_CORES_DN", "64")), kb)
        k = config.top_k
        up_gate = _EXPERT_MM["gu"](hidden_states, weights.gate_up_proj, top_k_indices, k, True)
        down_input = swiglu(up_gate)
        up_gate.deallocate(True)
        down = _EXPERT_MM["dn"](down_input, weights.down_proj, top_k_indices, k, False)
        down_input.deallocate(True)
        return down
    from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config, expert_decode_compute_kernel_config
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu

    k, H = config.top_k, config.hidden_size
    intermediate_size = weights.intermediate_size_per_device
    idx = ttnn.to_layout(top_k_indices, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(1, 2 * intermediate_size, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_GATE_UP", "1")))
    down_config = _build_sparse_matmul_config(1, H, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_DOWN", "1")))
    up_gate = ttnn.sparse_matmul(
        hidden_states, weights.gate_up_proj, sparsity=sparsity, indices=idx, memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile, program_config=gate_up_config, compute_kernel_config=expert_decode_compute_kernel_config(),
        dtype=ttnn.bfloat16,
    )
    up_gate = ttnn.reshape(up_gate, (1, k, 1, up_gate.shape[-1]))
    down_input = swiglu(up_gate) if swiglu is not None else apply_swiglu(up_gate)
    up_gate.deallocate(True)
    down = ttnn.sparse_matmul(
        down_input, weights.down_proj, sparsity=sparsity, indices=idx, memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile, program_config=down_config, compute_kernel_config=expert_decode_compute_kernel_config(),
        is_input_a_sparse=True, dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    if idx is not top_k_indices:
        ttnn.deallocate(idx)
    return down


def install_combine(layers, mesh):
    """DRAMA3_MOE_FUSED_COMBINE=1: the one-token MoE block's routed-slot sum and the gated shared expert add in one
    kernel (MoECombine) instead of slice + mul + fill-pad + sum + sigmoid + mul + add."""
    if os.environ.get("DRAMA3_MOE_FUSED_COMBINE", "0") != "1":
        return 0
    mc = MoECombine(mesh)
    # DRAMA3_MOE_FUSED_SWIGLU=1 (with the combine): the routed SwiGLU and the shared expert ([w3 | w1] + SwiGLU kernel)
    fused_sw = os.environ.get("DRAMA3_MOE_FUSED_SWIGLU", "0") == "1"
    if fused_sw:  # a tuned one-token program for the merged shared [w3 | w1] (as tt/decode_linear.py's (2048, 512) one)
        from models.demos.blackhole.drama3.tt import decode_linear

        decode_linear.TABLE.setdefault(((2048, 1024), "BFLOAT4_B", None), (2, 16, 2))
    sw_routed = Swiglu(mesh, int(os.environ.get("DRAMA3_MOE_SWIGLU_CORES", "64"))) if fused_sw else None
    sw_shared = Swiglu(mesh, 16) if fused_sw else None
    n = 0
    for layer in layers:
        moe = layer.feed_forward
        if getattr(moe, "shared", None) is None or not getattr(moe.experts, "indexed_decode", False):
            continue
        orig = moe.forward
        shared_fused = SharedExpertFused(moe.shared.mlp, mesh, sw_shared) if fused_sw else None

        def forward(x, mode="decode", _m=moe, _orig=orig, _sf=shared_fused):
            if mode == "decode" and x.shape[-2] > 1 and _m.config.num_devices == 1 and batch_on():
                return forward_batch(_m, x, _sf)
            if mode != "decode" or x.shape[-2] != 1 or _m.config.num_devices != 1:
                return _orig(x, mode=mode)
            H = x.shape[-1]
            x4 = ttnn.reshape(x, (1, 1, 1, H))
            x3 = ttnn.reshape(x, (1, 1, H))
            vals, idx = _m.router.topk(x4)
            if _sf is not None and shared_in_emm() and idx.layout == ttnn.ROW_MAJOR_LAYOUT:
                down = _down_shared(x4, idx, _m.experts.weights, _m.config, _sf)
                g = ttnn.linear(x3, _m.shared.gate_weight)
                out = mc(down, vals, None, g, shared_in_down=True)
                for t in (down, g, vals, idx):
                    ttnn.deallocate(t)
                return ttnn.reshape(out, tuple(x.shape))
            down = _down_indexed(x4, idx, _m.experts.weights, _m.config, _m.experts.decode_sparsity, sw_routed)
            g = ttnn.linear(x3, _m.shared.gate_weight)
            sh = _sf(x3) if _sf is not None else _m.shared.mlp.forward(x3)
            out = mc(down, vals, sh, g)
            for t in (down, g, sh, vals, idx):
                ttnn.deallocate(t)
            return ttnn.reshape(out, tuple(x.shape))

        moe.forward = forward
        n += 1
    _BATCH["sw"] = sw_routed
    return n


# ---------------------------------------------------------------- port6: B decode rows
_BATCH = {}


def batch_on():
    """DRAMA3_MOE_FUSED_BATCH=1 (port6, opt-in, with the fused top-k / combine / SwiGLU): the MoE block of B
    decode rows (batch mode) through RouterTopKB, the batched gather sparse_matmuls and MoECombineB."""
    return os.environ.get("DRAMA3_MOE_FUSED_BATCH", "0") == "1"


class RouterTopKB:
    """kernels/router_topk_b.cpp: one core per row: scores [1, 1, B, E] -> (top-k scores [1, 1, B, k] TILE,
    ids [1, 1, 1, B * k] uint16 ROW_MAJOR: the gather list of all rows), both fresh L1 tensors."""

    def __init__(self, mesh, E=256, k=8):
        self.mesh, self.E, self.k = mesh, E, k
        g = mesh.compute_with_storage_grid_size()
        self.g = g

    def __call__(self, scores):
        B = scores.shape[-2]
        L1 = ttnn.L1_MEMORY_CONFIG
        val = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, self.k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)
        idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, B * self.k]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, L1)
        cores = [ttnn.CoreCoord(i % self.g.x, i // self.g.x) for i in range(B)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        ct = [self.E, self.k, B]
        for t in (scores, val, idx):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        for b, c in enumerate(cores):
            rt[c.x][c.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address(), b]
        kern = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/router_topk_b{'2' if B > 16 else ''}.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [ttnn.CBDescriptor(total_size=2048 * 2, core_ranges=crs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])]
        ttnn.generic_op([scores, val, idx], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
        return val, idx


class MoECombineB:
    """kernels/moe_combine_b_*.cpp: out[b] = sum_j v[b, j] * down[b * k + j][row b] + sigmoid(g_b) * shared[b] for B
    decode rows; down [1, B * k, 32, H] (the batched gather sparse_matmul output), v [1, 1, B, k], shared [1, B, H],
    g [1, B, 1] -> [1, 1, B, H]. MODE (DRAMA3_MOE_COMBINE_B_MODE, default 0): 0 = the FPU column-broadcast multiplies
    accumulate in DST, 1 = products added on the SFPU."""

    def __init__(self, mesh, H=2048, k=8, n_cores=None):
        n_cores = n_cores or int(os.environ.get("DRAMA3_MOE_COMBINE_B_CORES", "32"))
        self.mesh, self.H, self.k, self.HT = mesh, H, k, H // 32
        assert self.HT % n_cores == 0
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.per = self.HT // n_cores
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.mode = int(os.environ.get("DRAMA3_MOE_COMBINE_B_MODE", "0"))

    def __call__(self, down, v, shared, g):
        B = v.shape[-2]
        S = B * self.k
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        rct = [self.k, self.HT, B]
        for t in (down, v, shared, g):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            t0 = i * self.per
            rrt[c.x][c.y] = [t0, self.per, down.buffer_address(), v.buffer_address(), shared.buffer_address(), g.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("moe_combine_b_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("moe_combine_b_compute.cpp", [S, self.per, self.mode], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: 1, 1: S, 2: S, 3: 1, 4: 1, 5: 1, 16: 2}
        cbs = [
            ttnn.CBDescriptor(total_size=2048 * n, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
            for i, n in sizes.items()
        ]
        ttnn.generic_op([down, v, shared, g, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


def forward_batch(moe, x, shared_fused):
    """The MoE block for B decode rows x [1, B, H] (batch mode) -> [1, B, H]."""
    from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config, expert_decode_compute_kernel_config
    from models.demos.blackhole.qwen36.tt.moe.operations import apply_swiglu

    mesh = x.device()
    if "rk" not in _BATCH:
        _BATCH["rk"] = RouterTopKB(mesh)
        _BATCH["mc"] = MoECombineB(mesh)
    B, H = x.shape[-2], x.shape[-1]
    k = moe.config.top_k
    L1 = ttnn.L1_MEMORY_CONFIG
    x4 = ttnn.reshape(x, (1, 1, B, H))
    x3 = ttnn.reshape(x, (1, B, H))
    r = moe.router
    sc = ttnn.linear(x4, r.proj_weight, compute_kernel_config=r.compute_kernel_config)
    raw, idx = _BATCH["rk"](sc)
    sc.deallocate(True)
    v = ttnn.softmax(raw, dim=-1)
    raw.deallocate(True)
    w = moe.experts.weights
    I = w.intermediate_size_per_device
    if dedup_on():  # port10: the expert matmuls over the unique routed experts of the B rows (ExpertMMB)
        if "emb_gu" not in _BATCH:
            _BATCH["emb_gu"] = ExpertMMB(mesh, int(os.environ.get("DRAMA3_MOE_DEDUP_NG_GU", "8")))
            _BATCH["emb_dn"] = ExpertMMB(mesh, int(os.environ.get("DRAMA3_MOE_DEDUP_NG_DN", "4")))
            _BATCH["mc3"] = MoECombineB3(mesh)
        msk = active_mask(mesh, B)
        ug = _BATCH["emb_gu"](x4, w.gate_up_proj, idx, B * k, True, msk, k)  # [1, NSLOT, 32, 2I]
        sw = _BATCH.get("sw")
        act = sw(ug) if sw is not None else apply_swiglu(ug)
        ug.deallocate(True)
        down = _BATCH["emb_dn"](act, w.down_proj, idx, B * k, False, msk, k)  # [1, NSLOT, 32, H]
        act.deallocate(True)
        g = ttnn.linear(x3, moe.shared.gate_weight)  # [1, B, 1]
        sh = _shared_rows(shared_fused, x3) if shared_fused is not None else moe.shared.mlp.forward(x3)
        out = _BATCH["mc3"](down, v, sh, g, idx, msk)
        for t in (down, g, sh, v, idx):
            ttnn.deallocate(t)
        return ttnn.reshape(out, tuple(x.shape))
    ot = ttnn.Tile([32, 32])
    gu_cfg = _build_sparse_matmul_config(B, 2 * I, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_GATE_UP", "1")))
    dn_cfg = _build_sparse_matmul_config(B, H, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_DOWN", "1")))
    ckc = expert_decode_compute_kernel_config()
    sp = moe.experts.decode_sparsity
    ug = ttnn.sparse_matmul(x4, w.gate_up_proj, sparsity=sp, indices=idx, memory_config=L1, output_tile=ot,
                            program_config=gu_cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1,1,1,B*k,M,2I]
    ug = ttnn.reshape(ug, (1, B * k, ug.shape[-2], ug.shape[-1]))
    sw = _BATCH.get("sw")
    act = sw(ug) if sw is not None else apply_swiglu(ug)
    ug.deallocate(True)
    down = ttnn.sparse_matmul(act, w.down_proj, sparsity=sp, indices=idx, memory_config=L1, output_tile=ot,
                              program_config=dn_cfg, compute_kernel_config=ckc, is_input_a_sparse=True, dtype=ttnn.bfloat16)
    act.deallocate(True)
    ttnn.deallocate(idx)
    if down.shape[-2] != 32:
        down = ttnn.reshape(down, (1, B * k, down.shape[-2], H), (1, B * k, 32, H))
    g = ttnn.linear(x3, moe.shared.gate_weight)  # [1, B, 1]
    if shared_fused is not None:
        sh = _shared_rows(shared_fused, x3)
    else:
        sh = moe.shared.mlp.forward(x3)
    out = _BATCH["mc"](down, v, sh, g)
    for t in (down, g, sh, v):
        ttnn.deallocate(t)
    return ttnn.reshape(out, tuple(x.shape))


def dedup_on():
    """DRAMA3_MOE_BATCH_DEDUP=1 (port10, opt-in, with DRAMA3_MOE_FUSED_BATCH): batch-mode MoE expert matmuls once per
    unique routed expert of the B rows (all rows in its M = 32 tile) instead of once per (row, slot): ExpertMMB +
    MoECombineB3. The unique list is built on the card from the router's ids at run time (trace-safe)."""
    return os.environ.get("DRAMA3_MOE_BATCH_DEDUP", "0") == "1"


def active_mask(mesh, B):
    """The active-row mask of the dedup path ([1, 32] uint16 ROW_MAJOR in DRAM, persistent per B, all ones until
    set_active_rows): rows with 0 are left out of the unique experts."""
    m = _BATCH.setdefault("mask", {})
    if B not in m:
        m[B] = ttnn.from_torch(torch.ones(1, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    return m[B]


def set_active_rows(mesh, B, active):
    """port10 DRAMA3_MOE_DEDUP_MASK (host/cbatch.py): mark the continuous-batching slots that hold a job (outside any
    trace: the traced kernels read the mask buffer at its fixed address)."""
    m = torch.zeros(1, 32, dtype=torch.int32)
    m[0, : len(active)] = torch.tensor([1 if a else 0 for a in active], dtype=torch.int32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT), active_mask(mesh, B))


class ExpertMMB:
    """port10 ExpertMM v3 (kernels/expert_mm3_*.cpp): out[u] = A_u @ W[uq[u]] over the U unique experts of the B * K
    routed ids (first-occurrence order, U found on the card), 64 cores as NG column groups x SG slot groups (core
    (n0, s0): NT / NG output tiles of the slots s0, s0 + SG, ..). A: the shared [1, 1, B, K] token rows (gate_up) or
    [1, NSLOT, 32, K] per slot (down). -> [1, NSLOT, 32, N] bf16 TILE in L1 (slots >= U not written)."""

    def __init__(self, mesh, NG, n_cores=64):
        assert n_cores % NG == 0
        self.mesh, self.NG, self.SG = mesh, NG, n_cores // NG
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.kb = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_KB", "16"))
        self.D = int(os.environ.get("DRAMA3_MOE_DEDUP_D", "8"))

    def __call__(self, a, w, ids, S, shared_a, msk, k):
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32
        KB = min(self.kb, KT)
        assert KT % KB == 0 and NT % self.NG == 0 and S <= 256
        NTC, SG, D = NT // self.NG, self.SG, self.D
        C = 2 * D
        nslot = S
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, nslot, 32, NT * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        acc = lambda *ts: [x for t in ts for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            n0, s0 = (i % self.NG) * NTC, i // self.NG
            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), n0, s0, SG, msk.buffer_address()]
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), n0, s0, SG, msk.buffer_address()]
            rc[c.x][c.y] = [s0, SG]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        fid = getattr(ttnn.MathFidelity, os.environ.get("QWEN36_DECODE_MATH_FIDELITY", "LoFi"))
        kernels = [
            kd("expert_mm3_reader.cpp", [KT, NT, KB, int(shared_a), S, D, C, NTC] + acc(a, w, ids, msk) + [k], rr, ttnn.ReaderConfigDescriptor()),
            kd("expert_mm3_writer.cpp" if S <= 128 else "expert_mm4_writer.cpp", [KT, NT, KB, S, D, C, NTC] + acc(w, ids, out, msk) + [k], rw, ttnn.WriterConfigDescriptor()),
            kd("expert_mm3_compute.cpp", [KT, KB, int(shared_a), NTC], rc,
               ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        wt = str(w.dtype).split(".")[-1]
        max_jobs = -(-nslot // SG) * NTC
        cb = lambda i, n, dt=ttnn.bfloat16, page=2048: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])
        cbs = [
            cb(0, KT if shared_a else 3 * KT),
            cb(1, C * KB // 2, w.dtype, TILE_BYTES[wt]),
            cb(4, C * KB // 2, w.dtype, TILE_BYTES[wt]),
            cb(2, 1),  # ids | pos table | unique list (512 bytes each)
            cb(5, 1),
            cb(6, 1, ttnn.bfloat16, 64),  # the unique count for the compute
            cb(16, max_jobs if S <= 128 else 16),  # B > 16: expert_mm4_writer drains it while the weights stream
        ]
        ttnn.generic_op([a, w, ids, msk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


class MoECombineB3:
    """port10 kernels/moe_combine_b3_*.cpp: MoECombineB over ExpertMMB's unique slots: out[b] = sum_j v[b, j] *
    down[pos[id[b k + j]]][row b] + sigmoid(g_b) * shared[b]; down [1, NSLOT, 32, H], ids the router's uint16 row."""

    def __init__(self, mesh, H=2048, k=8, n_cores=None):
        n_cores = n_cores or int(os.environ.get("DRAMA3_MOE_COMBINE_B_CORES", "32"))
        self.mesh, self.H, self.k, self.HT = mesh, H, k, H // 32
        assert self.HT % n_cores == 0
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.per = self.HT // n_cores
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, down, v, shared, g, ids, msk):
        B = v.shape[-2]
        S = B * self.k
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        rct = [self.k, self.HT, B]
        for t in (down, v, shared, g, ids, msk):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            t0 = i * self.per
            rrt[c.x][c.y] = [t0, self.per, down.buffer_address(), v.buffer_address(), shared.buffer_address(), g.buffer_address(), ids.buffer_address(), msk.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        ver = 3 if S <= 128 else 4  # B > 16: the down tiles in chunks of 16 (b4)
        kernels = [
            kd(f"moe_combine_b{ver}_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd(f"moe_combine_b{ver}_compute.cpp", [S, self.per], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: (1, 2048), 1: (S, 2048), 2: (S if ver == 3 else 32, 2048), 3: (1, 2048), 4: (1, 2048), 5: (1, 2048), 6: (1, 64), 7: (1, 2048), 16: (2, 2048)}
        cbs = [
            ttnn.CBDescriptor(total_size=page * n, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)])
            for i, (n, page) in sizes.items()
        ]
        ttnn.generic_op([down, v, shared, g, ids, msk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


def _shared_rows(sf, x3):
    """SharedExpertFused for B rows: [w3 | w1] matmul, the SwiGLU kernel on [1, 1, B, 2I], w2."""
    L1 = ttnn.L1_MEMORY_CONFIG
    B = x3.shape[-2]
    gu = ttnn.linear(x3, sf.w_gu, compute_kernel_config=sf.ckc, memory_config=L1)
    h = sf.sw(ttnn.reshape(gu, (1, 1, B, gu.shape[-1])))
    ttnn.deallocate(gu)
    out = ttnn.linear(ttnn.reshape(h, (1, B, h.shape[-1])), sf.w2, compute_kernel_config=sf.ckc, memory_config=L1)
    ttnn.deallocate(h)
    return out


class Swiglu:
    """out = up * silu(gate) over [up | gate] for one token per slot (kernels/swiglu_*.cpp): x [1, S, 1, 2N] bf16 TILE
    -> [1, S, 1, N], S * N/32 output tiles over n_cores cores (each the same count)."""

    def __init__(self, mesh, n_cores):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, x):
        S, N2 = x.shape[1], x.shape[-1]
        NT = N2 // 64
        items = S * NT
        per = items // len(self.cores)
        assert per * len(self.cores) == items, (items, len(self.cores))
        assert x.shape[2] <= 32  # one tile row per slot (B decode rows: rows 0..B-1)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, S, x.shape[2], N2 // 2]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rrt[c.x][c.y] = [i * per, per, x.buffer_address()]
            wrt[c.x][c.y] = [i * per, per, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kd("swiglu_reader.cpp", [NT] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()), rrt, ttnn.ReaderConfigDescriptor()),
            kd("swiglu_writer.cpp", list(ttnn.TensorAccessorArgs(out).get_compile_time_args()), wrt, ttnn.WriterConfigDescriptor()),
            kd("swiglu_compute.cpp", [per], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048 * per,
                core_ranges=self.crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i in (0, 1, 16)
        ]
        ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


class SharedExpertFused:
    """The shared expert's one-token MLP with w3 | w1 as ONE [H, 2I] matmul and the SwiGLU kernel
    (instead of w1 + silu, w3, mul): silu(w1 x) * (w3 x) -> w2."""

    def __init__(self, mlp, mesh, sw):
        w = mlp.weights
        gu = torch.cat([ttnn.to_torch(w.w3).float(), ttnn.to_torch(w.w1).float()], dim=-1)  # [up | gate]
        self.w_gu = ttnn.from_torch(gu.reshape(gu.shape[-2], gu.shape[-1]), dtype=w.w1.dtype, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.w2, self.ckc, self.sw = w.w2, mlp.compute_kernel_config_decode, sw

    def __call__(self, x3):
        L1 = ttnn.L1_MEMORY_CONFIG
        gu = ttnn.linear(x3, self.w_gu, compute_kernel_config=self.ckc, memory_config=L1)
        h = self.sw(ttnn.reshape(gu, (1, 1, 1, gu.shape[-1])))
        ttnn.deallocate(gu)
        out = ttnn.linear(ttnn.reshape(h, (1, 1, h.shape[-1])), self.w2, compute_kernel_config=self.ckc, memory_config=L1)
        ttnn.deallocate(h)
        return out


TILE_BYTES = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}


class ExpertMM:
    """One-token indexed expert matmul (kernels/expert_mm_*.cpp), DRAMA3_MOE_EXPERT_MM=1: out[j] = A_j @ W[ids[j]] for
    the k routed slots, the k * N/32 output tiles (jobs) spread over n_cores cores (sparse_matmul's program puts one
    output tile column per core: 32 cores for the gate_up of N = 1024). A: one shared row (gate_up) or one row per slot
    (down); ids: the router top-k's uint16 ROW_MAJOR ids. -> [1, k, 1, N] bf16 TILE (row 0 of each slot's tiles)."""

    def __init__(self, mesh, n_cores, kb=16):
        self.mesh, self.kb = mesh, kb
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, a, w, ids, nslot, shared_a, ws=None):
        """ws (port10, v2 only): the shared expert's weights [1, 1, K, N] as slot nslot - 1 (kernels/expert_mm5_*)."""
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32
        KB = min(self.kb, KT)
        assert KT % KB == 0
        # port8 DRAMA3_MOE_EXPERT_MM_V=2: D weight blocks in flight per RISC, a ring of C = 2 D blocks, outputs at the end
        v2 = os.environ.get("DRAMA3_MOE_EXPERT_MM_V", "1") == "2"
        D = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_D", "4"))
        C = 2 * D
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, nslot, 1, NT * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        jobs, C = nslot * NT, len(self.cores)
        per, extra = divmod(jobs, C)
        acc = lambda *ts: [x for t in ts for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        j0 = 0
        for i, c in enumerate(self.cores):
            nj = per + (1 if i < extra else 0)
            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), j0, nj] + ([ws.buffer_address()] if ws is not None else [])
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), j0, nj] + ([ws.buffer_address()] if ws is not None else [])
            rc[c.x][c.y] = [nj, j0] if v2 else [nj]
            j0 += nj
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        fid = getattr(ttnn.MathFidelity, os.environ.get("QWEN36_DECODE_MATH_FIDELITY", "LoFi"))
        dc = [D, C] if v2 else []
        sfx = "2" if v2 else ""
        if ws is not None:
            assert v2 and ws.dtype == w.dtype and (ws.shape[-2], ws.shape[-1]) == (w.shape[-2], w.shape[-1]), (ws.shape, w.shape, ws.dtype, w.dtype)
        rsrc, wsrc = (f"expert_mm{sfx}_reader.cpp", f"expert_mm{sfx}_writer.cpp") if ws is None else ("expert_mm5_reader.cpp", "expert_mm5_writer.cpp")
        wl = [w] if ws is None else [w, ws]
        kernels = [
            kd(rsrc, [KT, NT, KB, int(shared_a), nslot] + dc + acc(a, *wl, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd(wsrc, [KT, NT, KB, nslot] + dc + acc(*wl, ids, out), rw, ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/expert_mm{sfx}_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs, compile_time_args=[KT, KB, int(shared_a)] + ([NT] if v2 else []), runtime_args=rc,
                defines=[("EMM_NO_MATH", "1")] if os.environ.get("DRAMA3_MOE_EXPERT_MM_NO_MATH") else [],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False),
            ),
        ]
        wt = str(w.dtype).split(".")[-1]
        cbs = [
            ttnn.CBDescriptor(total_size=2048 * (KT if shared_a else (3 if v2 else 2) * KT), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)]),
            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * (C * KB // 2 if v2 else KB), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * (C * KB // 2 if v2 else KB), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=4, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
            ttnn.CBDescriptor(total_size=64, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=5, data_format=ttnn.bfloat16, page_size=64)]),
            ttnn.CBDescriptor(total_size=64, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=64)]),
            ttnn.CBDescriptor(total_size=2048 * (max(2, per + 1) if v2 else 2), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
            ttnn.CBDescriptor(total_size=2048 * max(1, KT // 16), core_ranges=self.crs,  # staging for the shared row's chunks
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=3, data_format=ttnn.bfloat16, page_size=2048)]),
        ]
        ttnn.generic_op([a, *wl, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out
