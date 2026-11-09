# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent / chunked DeltaNet forward (the `forward` dispatch).

Behavior-preserving extraction of the original `Qwen36GatedDeltaNet.forward` body.
Operates on the gdn instance: reads weights from `gdn.weights`, config dims from
`gdn.cfg`, mirrored scalar attrs + runtime state from `gdn`. Every ttnn op,
memory_config, and the `gated_deltanet_forward_ttnn` kwargs are verbatim.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, split_fused_conv_state
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as _gdn_mod
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn

# ep2 (QWEN36_SD_CONV_MM=1): single-device decode conv without the dim-1 slices of the fused [B, K-1, D] conv state.
# The stock path slices each state row (untilize -> slice -> tilize, 3x) and rebuilds the state by concat (untilize x2,
# concat, tilize); in the traced decode replay each of those programs waits ~40 us. Here every row is picked out, and
# the shifted state built, by tiny 0/1 selection matmuls in TILE layout (HiFi4, fp32 accumulate: a copy, bit-exact);
# the tap multiply/add sequence is unchanged, so the conv output is the same as the stock path's.
_SEL = {}


def _sel_mats(device, K):
    key = (id(device), K)
    if key not in _SEL:
        mats = []
        for k in range(K - 1):  # E_k [1,1,K-1]: row 0 picks state row k
            e = torch.zeros(1, 1, K - 1)
            e[0, 0, k] = 1.0
            mats.append(e)
        sh = torch.zeros(1, K - 1, K - 1)  # S: new row i = old row i+1
        for i in range(K - 2):
            sh[0, i, i + 1] = 1.0
        pl = torch.zeros(1, K - 1, 1)  # P: new last row = x
        pl[0, K - 2, 0] = 1.0

        def dev(t):
            return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        _SEL[key] = ([dev(e) for e in mats], dev(sh), dev(pl))
    return _SEL[key]


_SEL_CKC = None


def _conv_t1_inplace_mm(x, conv_buffer, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None):
    global _SEL_CKC
    if _SEL_CKC is None:
        _SEL_CKC = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
    mc = memory_config
    E, S, P = _sel_mats(device, kernel_size)
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = ttnn.matmul(E[k], conv_buffer, memory_config=mc, compute_kernel_config=_SEL_CKC)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        ttnn.deallocate(s_k)
        out = ttnn.add(out, term, memory_config=mc)
    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)
    shifted = ttnn.matmul(S, conv_buffer, memory_config=mc, compute_kernel_config=_SEL_CKC)
    placed = ttnn.matmul(P, x, memory_config=mc, compute_kernel_config=_SEL_CKC)
    new_state = ttnn.add(shifted, placed, memory_config=mc)
    ttnn.deallocate(shifted)
    ttnn.deallocate(placed)
    ttnn.copy(new_state, conv_buffer)
    ttnn.deallocate(new_state)
    return ttnn.silu(out, memory_config=mc), conv_buffer


if os.environ.get("QWEN36_SD_CONV_MM") == "1":
    _gdn_mod._causal_conv1d_decode_t1_inplace = _conv_t1_inplace_mm


def recurrent_forward(gdn, x, mode="recurrent", chunk_size=None, valid_len=None):
    """Non-kernel GDN forward. mode='chunk' (prefill, may delegate to the prefill kernel) or
    'recurrent' (single-token decode). Reads weights/state/dims off the gdn instance; updates
    gdn's recurrent + conv state in place or by reassignment per the trace-capture flags.

    valid_len: for fixed-bucket masked prefill — x is right-padded to T but only the first
    valid_len positions are real (see gated_deltanet_forward_ttnn). None = no padding."""
    w = gdn.weights
    if chunk_size is None:
        chunk_size = gdn.long_prefill_chunk_size if mode == "chunk" else 64

    if gdn.recurrent_state is None:
        shape = x.shape
        batch_size = shape[0] if len(shape) == 3 else 1
        init_recurrent_state(gdn, batch_size)

    T = x.shape[1]

    # After prefill, fuse separate conv states into one for efficient decode
    if T == 1 and gdn.fused_conv_state is None and gdn.conv_state_q is not None:
        gdn.fused_conv_state = ttnn.concat([gdn.conv_state_q, gdn.conv_state_k, gdn.conv_state_v], dim=2)
        gdn.fused_conv_state = ttnn.to_layout(gdn.fused_conv_state, ttnn.TILE_LAYOUT)
        split_fused_conv_state(gdn)

    # ep2: QWEN36_GDN_FUSED_DECODE=1 -> the one-token mixer body as one fused program (state updated in place)
    fz = getattr(gdn, "_fused1", None)
    if (
        fz is not None
        and mode == "recurrent"
        and T == 1
        and x.shape[0] == 1
        and fz.ready()
        and fz.ok(x)
    ):
        if not getattr(gdn, "_fused_logged", False):
            gdn._fused_logged = True
            from loguru import logger

            logger.info(f"ep2 QWEN36_GDN_FUSED_DECODE: fused 1-card GDN decode active (CMODE {fz.layout[3]})")
        return fz.forward(x, gdn.compute_kernel_config_decode)

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32, chunk_size=128).
    seq_masks = w.chunk_seq_masks_long

    output, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv = gated_deltanet_forward_ttnn(
        hidden_states=x,
        q_proj_weight=w.q_proj_weight,
        k_proj_weight=w.k_proj_weight,
        v_proj_weight=w.v_proj_weight,
        a_proj_weight=w.a_proj_weight,
        b_proj_weight=w.b_proj_weight,
        o_proj_weight=w.o_proj_weight,
        q_conv_weight=w.q_conv_weight,
        k_conv_weight=w.k_conv_weight,
        v_conv_weight=w.v_conv_weight,
        q_conv_bias=w.q_conv_bias,
        k_conv_bias=w.k_conv_bias,
        v_conv_bias=w.v_conv_bias,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        o_norm_weight=w.o_norm_weight,
        g_proj_weight=w.g_proj_weight,
        num_heads=gdn.num_heads,
        num_v_heads=gdn.num_v_heads,
        head_k_dim=gdn.head_k_dim,
        head_v_dim=gdn.head_v_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        use_gate=True,
        norm_eps=gdn.norm_eps,
        device=gdn.device,
        recurrent_state=gdn.recurrent_state,
        conv_state_q=gdn.conv_state_q,
        conv_state_k=gdn.conv_state_k,
        conv_state_v=gdn.conv_state_v,
        mode=mode,
        chunk_size=chunk_size,
        q_weight_taps=w.q_weight_taps,
        k_weight_taps=w.k_weight_taps,
        v_weight_taps=w.v_weight_taps,
        q_bias_dev=w.q_bias_dev,
        k_bias_dev=w.k_bias_dev,
        v_bias_dev=w.v_bias_dev,
        qkv_proj_weight=w.qkv_proj_weight,
        q_dim=gdn.cfg.q_dim,
        k_dim=gdn.cfg.k_dim,
        compute_kernel_config=gdn.compute_kernel_config_decode if mode == "recurrent" else gdn.compute_kernel_config,
        A_neg_precomputed=w.A_neg,
        fused_conv_weight_taps=w.fused_conv_weight_taps,
        fused_conv_bias_dev=w.fused_conv_bias_dev,
        fused_conv_state=gdn.fused_conv_state,
        fused_conv_state_split=getattr(gdn, "split_conv_state", None),
        ab_proj_weight=w.ab_proj_weight,
        mega_fused_weight=w.mega_fused_weight,
        mega_qkv_dim=w.mega_qkv_dim,
        mega_a_dim=w.mega_a_dim,
        mega_b_dim=w.mega_b_dim,
        mega_g_dim=w.mega_g_dim,
        use_inplace_state=gdn.use_inplace_state,
        chunk_seq_masks=seq_masks,
        valid_len=valid_len,
    )

    if gdn._chunk_inplace_state and mode == "chunk":
        # Per-chunk traced-prefill replay: write state into the persistent external
        # buffers in place so it carries across execute_trace() calls. gdn.recurrent_state
        # and gdn.fused_conv_state keep pointing at the same (baked) buffer addresses.
        if list(new_state.shape) != list(gdn.recurrent_state.shape):
            new_state = ttnn.reshape(new_state, list(gdn.recurrent_state.shape))
        ttnn.copy(new_state, gdn.recurrent_state)
        ttnn.deallocate(new_state)
        if new_fused_conv is not None and not isinstance(new_fused_conv, list):
            if new_fused_conv.layout != ttnn.TILE_LAYOUT:
                new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
            ttnn.copy(new_fused_conv, gdn.fused_conv_state)
            ttnn.deallocate(new_fused_conv)
        return output

    gdn.recurrent_state = new_state
    if isinstance(new_fused_conv, list):
        gdn.split_conv_state = new_fused_conv
    elif new_fused_conv is not None:
        gdn.fused_conv_state = new_fused_conv
    else:
        gdn.conv_state_q = new_conv_q
        gdn.conv_state_k = new_conv_k
        gdn.conv_state_v = new_conv_v
    return output
