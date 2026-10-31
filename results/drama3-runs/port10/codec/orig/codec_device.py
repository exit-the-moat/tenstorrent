# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fish's modded-DAC decoder (the codec's conv stack, 6.4 of its 6.8 GFLOP per frame) on the P150 (Day 4b).

The quantizer decode (codes -> 86 Hz latent z, 0.3 GFLOP per frame) stays on the host CPU. Layout on the card:
time-major [1, L, C] (rows = samples, columns = channels), BF16.
- causal dilated conv (CausalConvNet, k=7, stride 1): ttnn.conv1d with symmetric padding (k-1)*d and the first
  L output rows kept, i.e. x left-padded by (k-1)*d: exactly the causal conv;
- causal transposed conv (CausalTransConvNet, k=2s, stride s, right unpad s): a k=2 conv1d to s*Cout channels
  (tap 0 = x[t-1] with W[..., r+s], tap 1 = x[t] with W[..., r]), padding 1, first L rows, then the row-major
  reshape [L, s*Cout] -> [L*s, Cout] (time interleave);
- Snake1d: x + 1/(alpha + 1e-9) * sin(alpha * x)^2 with per-channel alpha rows;
- ResidualUnit: x + conv1(snake(conv7_d(snake(x)))).
Every conv runs at a fixed input length (one window size), so all programs compile once in warm-up.
"""

import os

import torch

import ttnn

# activations between ops: fp32 by default (audio SNR vs the host fp32 decoder 31.2 dB, 94 ms per 96-frame
# window); DRAMA3_CODEC_DTYPE=bfloat16: 28.4 dB, 56 ms
ACT = getattr(ttnn, os.environ.get("DRAMA3_CODEC_DTYPE", "float32"))
# DRAMA3_CODEC_WDTYPE=float32: fp32 conv / pointwise weights (default bfloat16)
WDT = getattr(ttnn, os.environ.get("DRAMA3_CODEC_WDTYPE", "bfloat16"))
# conv / matmul inputs rounded to bf16 on the card first (typecast rounds to nearest; the conv's own conversion
# of fp32 activations truncates: -0.31 % per k=7 conv). DRAMA3_CODEC_CONV_IN_BF16=0 feeds fp32 directly.
CONV_IN_BF16 = os.environ.get("DRAMA3_CODEC_CONV_IN_BF16", "1") == "1"
# snake intermediates in fp32 whatever the activation dtype (bf16 snakes add a +0.1 % level bias each)
SNAKE_FP32 = os.environ.get("DRAMA3_CODEC_SNAKE_FP32", "0") == "1"
# 2 passes per snake (multiply with sin + square post-activations, addcmul) instead of 5 (default;
# DRAMA3_CODEC_SNAKE_FUSED=0: the 5-op form). Snakes were 54 of the 111 ms (synced) of a window
SNAKE_FUSED = os.environ.get("DRAMA3_CODEC_SNAKE_FUSED", "1") == "1"


class _Conv:
    def __init__(self, w, b, k, d, pad, cin, cout, layout, mesh=None):
        self.k, self.d, self.pad, self.cin, self.cout, self.layout = k, d, pad, cin, cout, layout
        if k == 1:  # a pointwise conv is a matmul: [L, cin] @ [cin, cout] + b (conv1d's L1 plan fails at L=24576)
            self.w = ttnn.from_torch(w[:, :, 0].T.contiguous().float(), dtype=WDT, layout=ttnn.TILE_LAYOUT, device=mesh)
            self.b = ttnn.from_torch(b.reshape(1, -1).float(), dtype=WDT, layout=ttnn.TILE_LAYOUT, device=mesh)
            return
        # rounded to bf16 on the host (round to nearest): conv1d's own fp32 -> bf16 weight conversion truncates,
        # which shrank every k=7 conv by 0.31 % (an audio level error of -0.24 dB after the 24 convs)
        tdt = torch.bfloat16 if WDT == ttnn.bfloat16 else torch.float32
        self.w = ttnn.from_torch(w.float().contiguous().to(tdt), dtype=WDT)
        self.b = ttnn.from_torch(b.float().reshape(1, 1, 1, -1).to(tdt), dtype=WDT)


class DeviceDecoder:
    def __init__(self, mesh, codec, length):
        """codec: the host codec (fold_weight_norm()'d); length: latent rows per call (window frames * 4)."""
        self.mesh, self.L0 = mesh, length
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        m = codec.decoder.model
        self.stages = []  # ("conv", _Conv) | ("snake", (a, inv)) | ("trans", (_Conv, s, cout)) | ("res", [...]) | ("tanh", None)
        self.stages.append(("conv", self._causal(m[0].conv)))
        for blk in list(m)[1:5]:
            snake, trans, *res = list(blk.block)
            self.stages.append(("snake", self._snake(snake)))
            self.stages.append(("trans", self._trans(trans)))
            for ru in res:
                s1, c7, s2, c1 = list(ru.block)
                self.stages.append(("res", [self._snake(s1), self._causal(c7.conv), self._snake(s2), self._causal(c1.conv)]))
        self.stages.append(("snake", self._snake(m[5])))
        self.stages.append(("conv", self._causal(m[6].conv, pad_out_to=32)))
        self.stages.append(("tanh", None))

    # ---------------------------------------------------------------- weights
    @staticmethod
    def _layout(cin, cout):
        return ttnn.TensorMemoryLayout.BLOCK_SHARDED if max(cin, cout) >= 384 else ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    def _causal(self, conv, pad_out_to=None):
        w, b = conv.weight.detach(), conv.bias.detach()
        cout, cin, k = w.shape
        if pad_out_to and cout < pad_out_to:
            w = torch.cat([w, torch.zeros(pad_out_to - cout, cin, k)])
            b = torch.cat([b, torch.zeros(pad_out_to - cout)])
        d = conv.dilation[0]
        return _Conv(w, b, k, d, (k - 1) * d, cin, w.shape[0], self._layout(cin, w.shape[0]), self.mesh)

    def _trans(self, tc):
        w, b = tc.conv.weight.detach(), tc.conv.bias.detach()  # [Cin, Cout, 2s]
        cin, cout, k = w.shape
        s = tc.stride
        assert k == 2 * s
        wc = torch.empty(s * cout, cin, 2)
        for r in range(s):
            wc[r * cout : (r + 1) * cout, :, 1] = w[:, :, r].T
            wc[r * cout : (r + 1) * cout, :, 0] = w[:, :, r + s].T
        return _Conv(wc, b.repeat(s), 2, 1, 1, cin, s * cout, self._layout(cin, s * cout)), s, cout

    def _snake(self, sn):
        a = sn.alpha.detach().float().reshape(1, 1, -1)
        dt = ttnn.float32 if SNAKE_FP32 else ACT
        mk = lambda t: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh)  # noqa: E731
        return mk(a), mk((a + 1e-9).reciprocal())

    # ---------------------------------------------------------------- ops
    def _conv(self, x, L, c):
        """x [1, L, cin] (interleaved) -> [1, L, cout] TILE interleaved."""
        if c.k == 1:
            if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT)
        cfg = ttnn.Conv1dConfig(weights_dtype=WDT, shard_layout=c.layout, deallocate_activation=False)
        try:
            return self._conv_inner(x, L, c, cfg)
        except Exception as e:
            raise RuntimeError(f"conv1d L={L} cin={c.cin} cout={c.cout} k={c.k} d={c.d} layout={c.layout}: {str(e)[:200]}") from None

    def _conv_inner(self, x, L, c, cfg):
        if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        out, [c.w, c.b] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=c.w,
            bias_tensor=c.b,
            device=self.mesh,
            in_channels=c.cin,
            out_channels=c.cout,
            batch_size=1,
            input_length=L,
            kernel_size=c.k,
            stride=1,
            padding=c.pad,
            dilation=c.d,
            groups=1,
            dtype=ACT,
            conv_config=cfg,
            compute_config=self.ckc,
            return_weights_and_bias=True,
        )
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        n = 1
        for dsz in list(out.shape)[:-1]:
            n *= dsz
        out = ttnn.reshape(out, (1, n, out.shape[-1]))
        if n != L:
            out = ttnn.slice(out, [0, 0, 0], [1, L, out.shape[-1]])
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return out

    @staticmethod
    def _snake_op(x, ai):
        a, inv = ai
        if SNAKE_FP32 and x.dtype != ttnn.float32:
            xf = ttnn.typecast(x, ttnn.float32)
            t = ttnn.sin(ttnn.multiply(xf, a))
            t = ttnn.multiply(ttnn.multiply(t, t), inv)
            return ttnn.typecast(ttnn.add(xf, t), x.dtype)
        if SNAKE_FUSED:  # sin(a x)^2 as the multiply's fused post-activations, then x + inv * t in one addcmul
            t = ttnn.multiply(x, a, activations=[ttnn.UnaryOpType.SIN, ttnn.UnaryOpType.SQUARE])
            return ttnn.addcmul(x, t, inv, value=1.0)
        t = ttnn.multiply(x, a)
        t = ttnn.sin(t)
        t = ttnn.multiply(t, t)
        t = ttnn.multiply(t, inv)
        return ttnn.add(x, t)

    def _body(self, x, collect=False):
        L, outs = self.L0, []
        for kind, p in self.stages:
            if kind == "conv":
                x = self._conv(x, L, p)
            elif kind == "snake":
                x = self._snake_op(x, p)
            elif kind == "trans":
                c, s, cout = p
                y = self._conv(x, L, c)  # [1, L, s * cout]
                y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
                y = ttnn.reshape(y, (1, L * s, cout))
                L = L * s
                x = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
            elif kind == "res":
                s1, c7, s2, c1 = p
                y = self._snake_op(x, s1)
                y = self._conv(y, L, c7)
                y = self._snake_op(y, s2)
                y = self._conv(y, L, c1)
                x = ttnn.add(x, y)
            elif kind == "tanh":
                x = ttnn.tanh(x)
            if collect:
                outs.append((kind, ttnn.to_torch(x).float()))
        self.L_out = L
        return x, outs

    def __call__(self, z, collect=False):
        """z: latent [1, L0, 1024] (host fp32 or device) -> audio [L0 * 512] host fp32 (+ per-stage outputs)."""
        if getattr(self, "trace_id", None) is not None and isinstance(z, torch.Tensor) and not collect:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(z.float(), dtype=ACT, layout=ttnn.TILE_LAYOUT), self.z_buf)
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            return ttnn.to_torch(self.out).float()[0, : self.L_out, 0]
        if isinstance(z, torch.Tensor):
            z = ttnn.from_torch(z.float(), dtype=ACT, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        x, outs = self._body(z, collect)
        audio = ttnn.to_torch(x).float()[0, : self.L_out, 0]
        return (audio, outs) if collect else audio

    def capture(self):
        """Capture the decoder as one trace on a persistent latent buffer (after a warm-up call compiled every
        program). Capture it AFTER the generation traces: its persistent buffers then cannot sit in memory that
        one of those traces uses for intermediates."""
        self.z_buf = ttnn.from_torch(torch.zeros(1, self.L0, 1024), dtype=ACT, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        x, _ = self._body(self.z_buf)
        ttnn.deallocate(x)
        ttnn.synchronize_device(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out, _ = self._body(self.z_buf)
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)
