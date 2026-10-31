# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fish's quantizer decode (codes -> the codec latent of a stream window) on the P150 (port6 Track A, opt-in:
DRAMA3_QUANT_DEVICE=1). On the CPU it is the first window's ~50 ms latent wait (the post-module transformer alone
~34 ms for 4 frames: 8 layers of 13 M fp32 weights each, read for a handful of rows).

- the 10 codebook lookups + their 1x1 out_proj convs: host tables (host/quant_fast.py), summed on the CPU (< 1 ms)
  and uploaded as the window's [T, 1024] rows (T padded to the decoder window: 32 / 64 / 96 frames);
- the post-module (WindowLimitedTransformer, 8 layers: RMSNorm, wqkv, interleaved RoPE as x * C + (x @ P) * S,
  causal SDPA over the window (<= 96 frames, inside its 128 window), wo, layer scale, SwiGLU FFN, layer scale; final
  RMSNorm) and the upsample (2 x [causal transposed conv k = stride = 2 as one matmul + row interleave, ConvNeXt:
  causal depthwise k = 7 as one shift matmul + 7 weighted adds, LayerNorm, Linear, exact GELU, Linear, gamma,
  residual]) as ttnn ops, time-major [1, L, 1024], bf16 weights, fp32 accumulation.
The output z [1, 4 T, 1024] stays on the card and feeds the card decoder (tt/codec_device.py) directly. Rows past
the window's real frames are not zero (the padding codes run through the network), which the causal decoder never
reads for the real samples (the host path pads zeros; same samples kept).
"""

import os

import torch

import ttnn

from models.demos.blackhole.drama3.host import quant_fast


class DeviceQuantizer:
    def __init__(self, mesh, codec, sizes=(32, 64, 96)):
        q = codec.quantizer  # weight norm folded by the caller
        self.mesh = mesh
        self.traces = {}
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.tabs = [quant_fast.table(vq) for vq in q.semantic_quantizer.quantizers] + [quant_fast.table(vq) for vq in q.quantizer.quantizers]
        self.cs = [4096] + [1024] * 9
        dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.float().contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
        row = lambda v: dev(v.detach().float().reshape(1, 1, -1))  # noqa: E731
        pm = q.post_module
        assert isinstance(pm.input_proj, torch.nn.Identity) and isinstance(pm.output_proj, torch.nn.Identity)
        self.n_head, self.head_dim = pm.config.n_head, pm.config.head_dim
        self.eps = pm.config.norm_eps
        self.layers = []
        for blk in pm.layers:
            at, ff = blk.attention, blk.feed_forward
            self.layers.append(dict(
                n1=row(blk.attention_norm.weight), wqkv=dev(at.wqkv.weight.T), wo=dev(at.wo.weight.T),
                g1=row(blk.attention_layer_scale.gamma), n2=row(blk.ffn_norm.weight),
                w13=dev(torch.cat([ff.w1.weight.T, ff.w3.weight.T], -1)), w2=dev(ff.w2.weight.T),
                g2=row(blk.ffn_layer_scale.gamma),
            ))
        self.nf = row(pm.norm.weight)
        self.inter = pm.layers[0].feed_forward.w1.weight.shape[0]
        # interleaved RoPE (Fish: pairs (2i, 2i + 1) of each head, bf16 cos / sin tables): x * C + (x @ P) * S
        D = self.n_head * self.head_dim
        P = torch.zeros(D, D)
        P[torch.arange(1, D, 2), torch.arange(0, D, 2)] = -1.0  # out[2i] = -x[2i + 1]
        P[torch.arange(0, D, 2), torch.arange(1, D, 2)] = 1.0  # out[2i + 1] = x[2i]
        self.P = dev(P)
        fc = pm.freqs_cis  # [pos, head_dim / 2, 2] bf16
        self.rope = {}
        for T in sizes:
            c = fc[:T, :, 0].float().repeat_interleave(2, -1).repeat(1, self.n_head)  # [T, D]
            s = fc[:T, :, 1].float().repeat_interleave(2, -1).repeat(1, self.n_head)
            self.rope[T] = (dev(c.reshape(1, T, D)), dev(s.reshape(1, T, D)))
        # upsample stages
        self.ups = []
        L = None
        for stage in q.upsample:
            trans, blk = stage[0], stage[1]
            w = trans.conv.weight.detach().float()  # [Cin, Cout, k]
            cin, cout, k = w.shape
            wm = w.permute(0, 2, 1).reshape(cin, k * cout)  # column j * Cout + o: the row-major reshape interleaves
            dw = blk.dwconv.conv
            self.ups.append(dict(
                k=k, cout=cout, wt=dev(wm), bt=row(trans.conv.bias.detach().float().repeat(k)),
                dww=[row(dw.weight.detach().float()[:, 0, j]) for j in range(dw.weight.shape[-1])], dwb=row(dw.bias),
                ln_w=row(blk.norm.weight), ln_b=row(blk.norm.bias), ln_eps=blk.norm.eps,
                p1w=dev(blk.pwconv1.weight.T), p1b=row(blk.pwconv1.bias), p2w=dev(blk.pwconv2.weight.T), p2b=row(blk.pwconv2.bias),
                gamma=row(blk.gamma),
            ))
        self.dwk = len(self.ups[0]["dww"])
        self.shift = {}  # [7 L, L]: block j row t = x[t - 6 + j]
        for T in sizes:
            for L in (2 * T, 4 * T):
                if L in self.shift:
                    continue
                S = torch.zeros(self.dwk * L, L)
                for j in range(self.dwk):
                    d = self.dwk - 1 - j
                    S[j * L + torch.arange(d, L), torch.arange(0, L - d)] = 1.0
                self.shift[L] = dev(S)
        for T in sizes:  # compile every program
            ttnn.deallocate(self.forward_rows(torch.zeros(T, 1024)))

    # ---------------------------------------------------------------- ops
    def _lin(self, x, w, b=None):
        return ttnn.linear(x, w, bias=b, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.eps, compute_kernel_config=self.ckc)

    def _layer(self, x, p, T):
        D, H, hd = self.n_head * self.head_dim, self.n_head, self.head_dim
        h = self._rms(x, p["n1"])
        qkv = self._lin(h, p["wqkv"])  # [1, T, 3D]
        ttnn.deallocate(h)
        c, s = self.rope[T]
        outs = []
        for i in range(3):
            t = ttnn.slice(qkv, [0, 0, i * D], [1, T, (i + 1) * D])
            if i < 2:
                tp = ttnn.matmul(t, self.P, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
                t2 = ttnn.add(ttnn.multiply(t, c), ttnn.multiply(tp, s))
                ttnn.deallocate(tp)
                ttnn.deallocate(t)
                t = t2
            outs.append(ttnn.permute(ttnn.reshape(t, (1, T, H, hd)), (0, 2, 1, 3)))  # [1, H, T, hd]
        ttnn.deallocate(qkv)
        a = ttnn.transformer.scaled_dot_product_attention(outs[0], outs[1], outs[2], is_causal=True, scale=hd**-0.5,
                                                           compute_kernel_config=self.ckc)
        for t in outs:
            ttnn.deallocate(t)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), (1, T, D))
        o = self._lin(a, p["wo"])
        ttnn.deallocate(a)
        x = ttnn.add(x, ttnn.multiply(o, p["g1"]))
        ttnn.deallocate(o)
        h = self._rms(x, p["n2"])
        gu = self._lin(h, p["w13"])  # [w1 | w3]
        ttnn.deallocate(h)
        I = self.inter
        g = ttnn.slice(gu, [0, 0, 0], [1, T, I])
        u = ttnn.slice(gu, [0, 0, I], [1, T, 2 * I])
        ttnn.deallocate(gu)
        m = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        f = self._lin(m, p["w2"])
        ttnn.deallocate(m)
        y = ttnn.add(x, ttnn.multiply(f, p["g2"]))
        ttnn.deallocate(f)
        ttnn.deallocate(x)
        return y

    def _up(self, x, p, L):
        """x [1, L, C] -> [1, k L, C]."""
        y = self._lin(x, p["wt"], p["bt"])  # [1, L, k * C]
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.reshape(y, (1, L * p["k"], p["cout"]))
        x = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        L *= p["k"]
        # causal depthwise conv k = 7: the 7 shifted copies in one matmul, then the weighted sum
        X = ttnn.matmul(self.shift[L], ttnn.reshape(x, (L, p["cout"])), compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        y = None
        for j in range(self.dwk):
            t = ttnn.multiply(ttnn.reshape(ttnn.slice(X, [j * L, 0], [(j + 1) * L, p["cout"]]), (1, L, p["cout"])), p["dww"][j])
            y = t if y is None else ttnn.add(y, t)
        ttnn.deallocate(X)
        y = ttnn.add(y, p["dwb"])
        y = ttnn.layer_norm(y, epsilon=p["ln_eps"], weight=p["ln_w"], bias=p["ln_b"], compute_kernel_config=self.ckc)
        y = self._lin(y, p["p1w"], p["p1b"])
        y = ttnn.gelu(y, fast_and_approximate_mode=False)
        y = self._lin(y, p["p2w"], p["p2b"])
        y = ttnn.add(x, ttnn.multiply(y, p["gamma"]))
        ttnn.deallocate(x)
        return y, L

    def forward_rows(self, z0):
        """z0 [T, 1024] host (the summed codebook rows, T = a window size) -> z [1, 4 T, 1024] on the card (a traced
        size: this object's persistent output, do not free it; else a fresh tensor)."""
        T = z0.shape[0]
        h = ttnn.from_torch(z0.float().reshape(1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if T in self.traces:
            inp, tid, out = self.traces[T]
            ttnn.copy_host_to_device_tensor(h, inp)
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            return out
        return self._body(ttnn.to_device(h, self.mesh), free_input=True)

    def capture(self):
        """DRAMA3_QUANT_TRACE (default 1): every window size as a trace on a persistent input (call after the
        generation traces, as the card decoder's capture)."""
        if os.environ.get("DRAMA3_QUANT_TRACE", "1") != "1":
            return
        for T in self.rope:
            inp = ttnn.from_torch(torch.zeros(1, T, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            ttnn.deallocate(self._body(inp))
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out = self._body(inp)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self.traces[T] = (inp, tid, out)

    def _body(self, x, free_input=False):
        T = x.shape[-2]
        if not free_input:
            x = ttnn.clone(x)
        for p in self.layers:
            x = self._layer(x, p, T)
        x2 = self._rms(x, self.nf)
        ttnn.deallocate(x)
        x, L = x2, T
        for p in self.ups:
            x, L = self._up(x, p, L)
        return x

    def rows(self, codes, T):
        """codes [10, n] -> the summed table rows [T, 1024] (rows >= n zero), with Fish's index clamps."""
        n = codes.shape[1]
        z = torch.zeros(T, 1024)
        for i, tab in enumerate(self.tabs):
            z[:n] += tab[codes[i].clamp(0, self.cs[i] - 1)]
        return z

    def __call__(self, codes, T):
        return self.forward_rows(self.rows(codes.long(), T))
