# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""port6 Track A (opt-in, DRAMA3_QUANT_FAST=1): a faster CPU form of Fish's quantizer decode (codes -> the codec
latent of a stream window; the first window's latent is on the time-to-first-audio path). Same math, other op forms:
- each codebook's lookup + its 1x1 out_proj conv as ONE precomputed table [codebook size, 1024] (linear, so
  out_proj(codebook[i]) is a fixed row): 10 gathers + adds instead of 10 lookups + 10 convs;
- the causal transposed convs (k = stride = 2, no unpad) as one matmul [T, Cin] @ [Cin, Cout * 2] + interleave;
- the ConvNeXt depthwise causal convs (k = 7, groups = C) as 7 shifted multiply-adds.
The post-module transformer and the rest are unchanged. Not bit-identical (fp32 summation order).
"""

import types

import torch
import torch.nn.functional as F


def table(vq):
    """out_proj(codebook rows) of one VectorQuantize: [codebook size, 1024] fp32 (weight norm folded)."""
    w = vq.out_proj.weight.detach().float()  # [1024, 8, 1]
    b = vq.out_proj.bias.detach().float()
    cb = vq.codebook.weight.detach().float()  # [cs, 8]
    return (cb @ w[:, :, 0].T + b).contiguous()


def _table_from_codes(rvq):
    tabs = [table(vq) for vq in rvq.quantizers]

    def from_codes(self, codes):
        # codes [B, N, T] -> z [B, 1024, T]
        z = None
        for i in range(codes.shape[1]):
            r = F.embedding(codes[:, i, :], tabs[i])  # [B, T, 1024]
            z = r if z is None else z + r
        return z.transpose(1, 2), None, codes

    rvq.from_codes = types.MethodType(from_codes, rvq)


def _fast_transconv(m):
    conv = m.conv
    w = conv.weight.detach().float()  # [Cin, Cout, k]
    cin, cout, k = w.shape
    assert conv.stride[0] == k and m.kernel_size - m.stride == 0
    wm = w.reshape(cin, cout * k).contiguous()
    b = conv.bias.detach().float()

    def forward(self, x):
        B, C, T = x.shape
        y = (x.transpose(1, 2) @ wm).reshape(B, T, cout, k).permute(0, 2, 1, 3).reshape(B, cout, T * k)
        return (y + b[None, :, None]).contiguous()

    m.forward = types.MethodType(forward, m)


def _fast_dwconv(m):
    conv = m.conv
    assert conv.groups == conv.in_channels and conv.stride[0] == 1 and conv.dilation[0] == 1
    w = conv.weight.detach().float()[:, 0, :]  # [C, k]
    b = conv.bias.detach().float()
    k = w.shape[-1]
    assert m.padding == k - 1

    def forward(self, x):
        T = x.shape[-1]
        xp = F.pad(x, (k - 1, 0))
        y = b[None, :, None] + w[None, :, 0:1] * xp[:, :, 0:T]
        for j in range(1, k):
            y = y + w[None, :, j : j + 1] * xp[:, :, j : j + T]
        return y

    m.forward = types.MethodType(forward, m)


def install(q):
    """q: the codec's DownsampleResidualVectorQuantize (weight norm folded). Patches its decode path in place."""
    _table_from_codes(q.semantic_quantizer)
    _table_from_codes(q.quantizer)
    for stage in q.upsample:
        trans, block = stage[0], stage[1]
        _fast_transconv(trans)
        _fast_dwconv(block.dwconv)
    return q
