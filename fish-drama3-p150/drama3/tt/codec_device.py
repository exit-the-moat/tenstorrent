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
# port10 (opt-in): channel 0 (the only real one of the 32 padded output channels) taken on the card before the
# readback (transpose, slice, tanh on that row, untilize): 1/32 of the PCIe bytes
OUT_CH0 = os.environ.get("DRAMA3_CODEC_OUT_CH0", "0") == "1"
# port10 (opt-in): causal convs padded on the left only (conv1d padding (p, 0)): exactly L output rows, no slice
CAUSAL_PAD = os.environ.get("DRAMA3_CODEC_CAUSAL_PAD", "0") == "1"
# port10 (opt-in): Conv1dConfig knobs for the k > 1 convs, comma list: adb (act double buffer), wdb (weights
# double buffer), full (full_inner_dim), split (force_split_reader), reuse (activation reuse), abh=<n> (act_block_h).
# Measured at w = 32: split / full no change, wdb slower, adb wrong audio (SNR -0.5 dB); config_tensors_in_dram hung the card (port10), so it is not offered
CONV_OPTS = [o for o in os.environ.get("DRAMA3_CODEC_CONV_OPTS", "").split(",") if o]
# port10 (opt-in): each Snake1d as one fused kernel (kernels/snake_*.cpp) with the default path's LLK calls in DST
# (multiply + sin + square, addcmul, typecast to bf16 for the conv that reads it): one fp32 read and one bf16 write
# per element instead of three ops' DRAM round trips. Only with the default fp32 activations and bf16 conv inputs.
SNAKE_KERNEL = os.environ.get("DRAMA3_CODEC_SNAKE_KERNEL", "0") == "1"
# port10 (opt-in, with SNAKE_KERNEL): each residual add fused with the Snake1d that follows it (kernels/snake_add_*.cpp:
# the fp32 sum for the residual stream and the bf16 snake in one pass)
SNAKE_ADD = os.environ.get("DRAMA3_CODEC_SNAKE_ADD", "0") == "1"
# port10 precision candidate (opt-in): math fidelity of the convs / pointwise matmuls (default HiFi4)
FIDELITY = getattr(ttnn.MathFidelity, os.environ.get("DRAMA3_CODEC_FIDELITY", "HiFi4"))
KDIR = "models/demos/blackhole/drama3/kernels"


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
        self._sk = {}  # fused snake kernel: per-shape program parts
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=FIDELITY, fp32_dest_acc_en=True, packer_l1_acc=True
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
        inv = (a + 1e-9).reciprocal()
        # the fused kernel's operands: fp32 [1, 32, C], the row replicated (what binary_ng's row broadcast reads)
        mk32 = lambda t: ttnn.from_torch(t.expand(1, 32, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh)  # noqa: E731
        return mk(a), mk(inv), mk32(a), mk32(inv)

    # ---------------------------------------------------------------- ops
    def _conv(self, x, L, c):
        """x [1, L, cin] (interleaved) -> [1, L, cout] TILE interleaved."""
        if c.k == 1:
            if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT)
        cfg = ttnn.Conv1dConfig(weights_dtype=WDT, shard_layout=c.layout, deallocate_activation=False)
        for o in CONV_OPTS:
            if o == "adb":
                cfg.enable_act_double_buffer = True
            elif o == "wdb":
                cfg.enable_weights_double_buffer = True
            elif o == "full":
                cfg.full_inner_dim = True
            elif o == "split":
                cfg.force_split_reader = True
            elif o == "reuse":
                cfg.enable_activation_reuse = True
            elif o.startswith("abh="):
                cfg.act_block_h_override = int(o[4:])
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
            padding=(c.pad, 0) if CAUSAL_PAD else c.pad,
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

    def _snake_op(self, x, ai):
        a, inv = ai[:2]
        if SNAKE_KERNEL and x.dtype == ttnn.float32 and CONV_IN_BF16 and SNAKE_FUSED and not SNAKE_FP32:
            return self._snake_kernel(x, ai[2], ai[3])
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

    def _snake_kernel(self, x, a32, i32, y=None):
        """x [1, L, C] fp32 TILE interleaved -> snake(x) [1, L, C] bf16 TILE (DRAM), kernels/snake_*.cpp. With y
        (SNAKE_ADD): -> (x + y fp32, snake(x + y) bf16), kernels/snake_add_*.cpp."""
        L, C = x.shape[-2], x.shape[-1]
        add = y is not None
        key = (L, C, add)
        if key not in self._sk:  # per shape: cores, compile-time args, CBs, compute config
            R, CT = L // 32, C // 32
            items = R * CT
            g = self.mesh.compute_with_storage_grid_size()
            per = max(-(-items // (g.x * g.y)), min(8, items))  # >= 8 tiles per core (fewer runtime args to set)
            nc = -(-items // per)
            cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(nc)]
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            work = [(c, i * per, min(per, items - i * per)) for i, c in enumerate(cores)]
            ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
            dram = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
            if add:
                cts = ([R, CT] + ta(x) + ta(y) + ta(a32) + ta(i32), [R, CT] + ta(dram) + ta(dram), [R])
            else:
                cts = ([R, CT] + ta(x) + ta(a32) + ta(i32), [R, CT] + ta(dram), [R])
            ttnn.deallocate(dram)
            ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for i in (0, 1, 2, 3):
                modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ccfg.unpack_to_dest_mode = modes
            cb = lambda i, dt, pg, n: ttnn.CBDescriptor(  # noqa: E731
                total_size=pg * n, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)]
            )
            cbs = [cb(0, ttnn.float32, 4096, 4), cb(1, ttnn.float32, 4096, 2), cb(2, ttnn.float32, 4096, 2), cb(16, ttnn.bfloat16, 2048, 4)]
            if add:
                cbs += [cb(3, ttnn.float32, 4096, 4), cb(17, ttnn.float32, 4096, 4)]
            self._sk[key] = (crs, work, cts, ccfg, cbs)
        crs, work, cts, ccfg, cbs = self._sk[key]
        mk = lambda dt: ttnn.allocate_tensor_on_device(ttnn.Shape([1, L, C]), dt, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
        out = mk(ttnn.bfloat16)
        sm = mk(ttnn.float32) if add else None
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        xa, aa, ia, oa = x.buffer_address(), a32.buffer_address(), i32.buffer_address(), out.buffer_address()
        if add:
            ya, sa = y.buffer_address(), sm.buffer_address()
            for c, i0, n in work:
                rrt[c.x][c.y] = [i0, n, xa, ya, aa, ia]
                wrt[c.x][c.y] = [i0, n, oa, sa]
                crt[c.x][c.y] = [i0, n]
        else:
            for c, i0, n in work:
                rrt[c.x][c.y] = [i0, n, xa, aa, ia]
                wrt[c.x][c.y] = [i0, n, oa]
                crt[c.x][c.y] = [i0, n]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        pre = "snake_add" if add else "snake"
        kernels = [
            kd(f"{pre}_reader.cpp", cts[0], rrt, ttnn.ReaderConfigDescriptor()),
            kd(f"{pre}_writer.cpp", cts[1], wrt, ttnn.WriterConfigDescriptor()),
            kd(f"{pre}_compute.cpp", cts[2], crt, ccfg),
        ]
        io = [x, y, a32, i32, out, sm] if add else [x, a32, i32, out]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return (sm, out) if add else out

    def _body(self, x, collect=False):
        L, outs = self.L0, []
        fuse = SNAKE_ADD and SNAKE_KERNEL and ACT == ttnn.float32 and CONV_IN_BF16 and SNAKE_FUSED and not SNAKE_FP32 and not collect
        pend = None  # SNAKE_ADD: the residual branch output whose add is fused into the next snake
        for kind, p in self.stages:
            if pend is not None and kind not in ("snake", "res"):
                x, pend = ttnn.add(x, pend), None
            if kind == "conv":
                x = self._conv(x, L, p)
            elif kind == "snake":
                if pend is not None:
                    _, x = self._snake_kernel(x, p[2], p[3], y=pend)
                    pend = None
                else:
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
                if pend is not None:
                    x, y = self._snake_kernel(x, s1[2], s1[3], y=pend)
                    pend = None
                else:
                    y = self._snake_op(x, s1)
                y = self._conv(y, L, c7)
                y = self._snake_op(y, s2)
                y = self._conv(y, L, c1)
                if fuse and x.dtype == ttnn.float32 and y.dtype == ttnn.float32:
                    pend = y
                else:
                    x = ttnn.add(x, y)
            elif kind == "tanh":
                if OUT_CH0:  # [1, L, 32] -> row 0 of [1, 32, L], tanh, row major [1, 1, L]
                    x = ttnn.transpose(x, -2, -1)
                    x = ttnn.slice(x, [0, 0, 0], [1, 1, L])
                    x = ttnn.tanh(x)
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                else:
                    x = ttnn.tanh(x)
            if collect:
                outs.append((kind, ttnn.to_torch(x).float()))
        self.L_out = L
        return x, outs

    def audio(self, x):
        """The decoder's output tensor -> audio [L_out] host fp32."""
        t = ttnn.to_torch(x).float()
        return t.reshape(-1)[: self.L_out] if OUT_CH0 else t[0, : self.L_out, 0]

    def __call__(self, z, collect=False):
        """z: latent [1, L0, 1024] (host fp32 or device) -> audio [L0 * 512] host fp32 (+ per-stage outputs)."""
        if getattr(self, "trace_id", None) is not None and isinstance(z, torch.Tensor) and not collect:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(z.float(), dtype=ACT, layout=ttnn.TILE_LAYOUT), self.z_buf)
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            return self.audio(self.out)
        if isinstance(z, torch.Tensor):
            z = ttnn.from_torch(z.float(), dtype=ACT, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        x, outs = self._body(z, collect)
        audio = self.audio(x)
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
