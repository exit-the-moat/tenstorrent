# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 TTS, Day 5b: B voices at once on one P150 (opt-in batch mode; the one-row pipeline is generate_fast).

The B streams share the voice-prefix cache: the prefix K/V blocks are prefilled once and every stream's page table
points at them (read only), the GDN prefix state is restored per stream. Each line's suffix is prefilled with the
one-row prefill path (stream b's page-table row), its GDN state is copied into row b of the batched decode state,
and then all B streams decode together: one backbone trace and one tower trace (device codebook sampling) per
frame with B rows, the main-token draws on the host per stream (each with its own generator, the solo RNG order),
the card codec per stream between frames. A finished stream keeps its slot until the batch ends.

    run_gen.sh <log> --batch --batch-size 4 --lines-file /opt/ep0/voice/card_lines.json --out <dir>
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, log, reference_codes

BLOCK = 64


class BatchState:
    """Batched decode state + traces for B streams on top of a warmed FastSession (warmup(capture=False))."""

    def __init__(self, s, B):
        from models.demos.blackhole.drama3.tt import decode_linear, decode_norm
        from models.demos.blackhole.drama3.tt.device_sampler import K

        self.s, self.B = s, B
        bb, tw, mesh = s.bb, s.tower, s.mesh
        self.mesh = mesh
        assert int(os.environ.get("DRAMA3_KV_STREAMS", "1")) >= B, "set DRAMA3_KV_STREAMS >= B before the session"
        assert bb.rope_full and s.device_sampling and os.environ.get("DRAMA3_SHARED_POS", "1") == "1"
        nb = bb.max_seq // BLOCK
        nP = s.P // BLOCK
        self.nb = nb
        # page table row b: the shared prefix blocks, then stream b's own blocks of the pool (row 0 = identity)
        rows = [list(range(nP)) + [b * nb + j for j in range(nP, nb)] for b in range(B)]
        if os.environ.get("DRAMA3_KV_COMPACT", "0") == "1":  # port10: the pool holds the prefix blocks once
            kp = int(os.environ.get("DRAMA3_KV_PREFIX_BLOCKS", "10"))
            assert nP >= kp, ("the voice prefix is shorter than DRAMA3_KV_PREFIX_BLOCKS", nP, kp)
            rows = [list(range(nb))] + [list(range(nP)) + [nb + (b - 1) * (nb - kp) + (j - kp) for j in range(nP, nb)]
                                        for b in range(1, B)]
        self.table_host = torch.tensor(rows, dtype=torch.int32)
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)  # noqa: E731
        self.table = dev(self.table_host)
        self.row_tables = [dev(self.table_host[b : b + 1].contiguous()) for b in range(B)]
        self.solo_table = (bb.page_table, bb.page_table_host, tw.page_table, tw.page_table_host)
        # the attention / GDN mixers take [1, B, H] as B one-token sequences (their rows_are_batch attribute)
        self.mixers = [layer.attention for layer in bb.layers] + [layer.attn for layer in tw.shared + tw.residual]
        # batched GDN state (the one-row buffers stay the prefill's)
        z = lambda shape, dt=ttnn.bfloat16: ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import ConvPartialSums

        self.gdn = []
        # port6 DRAMA3_GDN_FUSED_BATCH=1: the fused v4 GDN decode keeps sequence b's partial sums in row 2b
        from models.demos.blackhole.drama3.tt.gdn_fused_decode import batch_on

        self.q_rows = 2 if (batch_on() and B > 1) else 1
        for dn, rec, conv in bb.gdn_state:
            rb = z([B] + list(rec.shape)[1:])
            cb = z([B] + list(conv.shape)[1:])
            parts = dn.split_conv_state
            assert parts is not None and bb.split_conv
            sp = [z([1, B * self.q_rows] + list(p.shape)[2:], p.dtype) for p in parts]  # [1, B, D]: one row per sequence
            sp = ConvPartialSums(sp) if getattr(parts, "partial_sums", False) else sp
            self.gdn.append((dn, rb, cb, sp))
        self.solo_gdn = [(dn.recurrent_state, dn.fused_conv_state, dn.split_conv_state) for dn, _, _ in bb.gdn_state]
        # decode inputs
        H = bb.args.dim
        self.x_buf = ttnn.from_torch(torch.zeros(1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.pos_buf = dev(torch.zeros(B, dtype=torch.int32))
        c, sn = self.cos_sin(torch.zeros(B, dtype=torch.long))
        self.cos_buf, self.sin_buf = ttnn.to_device(c, mesh), ttnn.to_device(sn, mesh)
        # tower inputs
        ns = len(tw.shared)
        self.n_host = ns + 1
        self.e_host_buf = ttnn.from_torch(torch.zeros(self.n_host, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.e_bufs = [
            ttnn.from_torch(torch.zeros(1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for _ in tw.shared + tw.residual
        ]
        self.noise_buf = ttnn.from_torch(torch.ones(1, len(tw.residual), B, K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.K = K
        self._rows_cfg = (decode_norm, decode_linear)

    # ---------------------------------------------------------------- mode switches
    def cos_sin(self, positions):
        bb = self.s.bb
        c = bb.rope.cos_cpu[positions].float()
        sn = bb.rope.sin_cpu[positions].float()
        pad = bb.args.head_dim - c.shape[-1]
        n = positions.shape[0]
        c = torch.cat([c, torch.ones(n, pad)], -1).reshape(n, 1, -1)
        sn = torch.cat([sn, torch.zeros(n, pad)], -1).reshape(n, 1, -1)
        return (
            ttnn.from_torch(c.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(sn.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        )

    def batch_mode(self, on):
        """on: the mixers take B rows, the GDN layers point at the batched state, one-tile norms / linears take
        B rows; off: everything as the one-row pipeline (prefill)."""
        dnorm, dlin = self._rows_cfg
        for m in self.mixers:
            m.rows_are_batch = on and self.B > 1
        dnorm.ROWS = dlin.ROWS = self.B if on else 1
        for (dn, rb, cb, sp), solo in zip(self.gdn, self.solo_gdn):
            dn.recurrent_state, dn.fused_conv_state, dn.split_conv_state = (rb, cb, sp) if on else solo
        tw = self.s.tower
        for layer in tw.shared + tw.residual:
            layer.page_table = self.table if on else tw.page_table

    def stream_table(self, b):
        """Prefill of stream b: its page-table row in the backbone and the tower (None: the one-row identity)."""
        bb, tw = self.s.bb, self.s.tower
        if b is None:
            bb.page_table, bb.page_table_host, tw.page_table, tw.page_table_host = self.solo_table
        else:
            bb.page_table, bb.page_table_host = self.row_tables[b], self.table_host[b : b + 1]
            tw.page_table, tw.page_table_host = self.row_tables[b], self.table_host[b : b + 1]
        for layer in tw.shared + tw.residual:
            layer.page_table = tw.page_table

    # ---------------------------------------------------------------- bodies
    def _bb_body(self):
        from models.tt_transformers.tt.common import Mode

        bb = self.s.bb
        x = self.x_buf
        for layer in bb.layers:
            if layer.is_full_attention:
                y = layer.forward(x, cos=self.cos_buf, sin=self.sin_buf, mode="decode", position_tensor=self.pos_buf, page_table=self.table)
            else:
                y = layer.forward(x, mode="decode")
            if len(y.shape) == 4:
                y = ttnn.reshape(y, (1, y.shape[-2], y.shape[-1]))
            if x is not self.x_buf:
                ttnn.deallocate(x)
            x = y
        xn = bb.norm(x, mode=Mode.PREFILL)
        ttnn.deallocate(x)
        logits = ttnn.linear(xn, bb.head, dtype=ttnn.float32, compute_kernel_config=bb.head_ckc)
        return xn, logits

    def _tw_body(self, h_in):
        tw = self.s.tower
        B, ns = self.B, len(tw.shared)
        for i in range(self.n_host):
            ttnn.copy(ttnn.slice(self.e_host_buf, [i, 0, 0], [i + 1, B, tw.H]), self.e_bufs[i])
        h, codes = h_in, []
        for i in range(len(tw.residual)):
            layers = [*tw.shared, tw.residual[0]] if i == 0 else [tw.residual[i]]
            bufs = self.e_bufs[: ns + 1] if i == 0 else [self.e_bufs[ns + i]]
            h0 = h
            for layer, e in zip(layers, bufs):
                y = layer(h, e, self.cos_buf, self.sin_buf, pos=self.pos_buf)
                if h is not h0:
                    ttnn.deallocate(h)
                h = y
            if h0 is not h_in:
                ttnn.deallocate(h0)
            lg = ttnn.linear(h, tw.heads[i], dtype=ttnn.float32, compute_kernel_config=tw.ckc)
            noise = ttnn.slice(self.noise_buf, [0, i, 0, 0], [1, i + 1, B, self.K])
            code, onehot = tw.sampler(ttnn.reshape(lg, (1, 1, B, lg.shape[-1])), noise)
            ttnn.deallocate(lg)
            codes.append(code)
            if i + 1 < len(tw.residual):
                e = ttnn.matmul(onehot, tw.res_emb[i + 1], compute_kernel_config=tw.ckc, dtype=ttnn.bfloat16)
                ttnn.copy(ttnn.reshape(e, (1, B, tw.H)), self.e_bufs[ns + i + 1])
        ttnn.deallocate(h)
        return ttnn.concat(codes, dim=-1)  # [1, 1, B, 9]

    def capture(self):
        """Compile runs, then the two traces (backbone, then tower reading the backbone's output in place)."""
        self.batch_mode(True)
        try:
            xn, lg = self._bb_body()
            ttnn.deallocate(lg)
            ttnn.deallocate(self._tw_body(xn))
            ttnn.deallocate(xn)
            ttnn.synchronize_device(self.mesh)
            self.bb_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.dec_out = self._bb_body()
            ttnn.end_trace_capture(self.mesh, self.bb_tid, cq_id=0)
            self.tw_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.codes_out = self._tw_body(self.dec_out[0])
            ttnn.end_trace_capture(self.mesh, self.tw_tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
        finally:
            self.batch_mode(False)

    # ---------------------------------------------------------------- state transfer
    def read_gdn(self):
        """The one-row GDN state (after a stream's prefill) -> host copies."""
        out = []
        for dn, _, _ in self.s.bb.gdn_state:
            out.append((ttnn.to_torch(dn.recurrent_state), [ttnn.to_torch(p) for p in dn.split_conv_state]))
        return out

    def load_gdn(self, per_stream):
        """Host copies of B streams' GDN states -> the batched buffers (one upload per buffer)."""
        for li, (dn, rb, cb, sp) in enumerate(self.gdn):
            rec = torch.cat([st[li][0] for st in per_stream], 0)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(rec, dtype=rb.dtype, layout=ttnn.TILE_LAYOUT), rb)
            for k, buf in enumerate(sp):
                part = torch.cat([st[li][1][k] for st in per_stream], 1)
                if self.q_rows == 2:  # sequence b in row 2b, row 2b + 1 zero
                    part = torch.stack([part, torch.zeros_like(part)], 2).reshape(part.shape[0], 2 * part.shape[1], -1)
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(part, dtype=buf.dtype, layout=ttnn.TILE_LAYOUT), buf)

    def stage(self, emb_rows, positions):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(emb_rows.reshape(1, self.B, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.x_buf
        )
        self.stage_pos(positions)

    def stage_pos(self, positions):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(positions.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.pos_buf
        )
        c, sn = self.cos_sin(positions)
        ttnn.copy_host_to_device_tensor(c, self.cos_buf)
        ttnn.copy_host_to_device_tensor(sn, self.sin_buf)


def _draw(tower, cand_logits, window, pre, temperature, top_p, top_k):
    """The host part of solo _step_fused_pre for one stream: main token (+ RAS) and the 5 host embedding rows."""
    from models.demos.blackhole.drama3.host.tower import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
    from models.demos.blackhole.drama3.tt.tower_device import sample_with_noise

    t = tower.host
    top_k = top_k or len(t.candidates)
    first = int(t.candidates[sample_with_noise(cand_logits, temperature, top_k, top_p, pre["n1"], keep_crossing=True)])
    ras = int(t.candidates[sample_with_noise(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, pre["n2"], keep_crossing=True)])
    main = ras if (first in window and t.sem0 <= first < t.sem0 + t.cb_size) else first
    code0 = min(max(main - t.sem0, 0), t.cb_size - 1)
    lab = t.is_semantic(torch.tensor([main]))
    c0 = torch.tensor([code0])
    rows = [tower._emb_rows(layer, c0, lab, pre["si"] if i == 0 else None) for i, layer in enumerate(tower.shared)]
    rows.append(tower._emb_rows(tower.residual[0], c0, lab))
    return main, code0, first != main, torch.cat(rows)  # [5, H]


def _sample_rows(L, temperature, top_k, top_p, N, m=64):
    """port10: tt/tower_device.sample_with_noise (keep_crossing=True) for n rows at once: L [n, C] logits, N [n, C]
    noise -> [n] chosen indices. Rows whose kept ranks may reach past the top m fall back to the one-row function."""
    from models.demos.blackhole.drama3.tt.tower_device import MIN_TEMPERATURE, sample_with_noise

    scaled = L.float() / max(temperature, MIN_TEMPERATURE)
    vals, ids = torch.topk(scaled, m, dim=-1)
    cum = torch.cumsum(torch.exp(vals - torch.logsumexp(scaled, -1, keepdim=True)), -1)
    drop = cum > top_p
    drop[:, 1:] = drop[:, :-1].clone()
    drop |= torch.arange(m) >= top_k
    drop[:, 0] = False
    probs = torch.softmax(vals.masked_fill(drop, float("-inf")), -1)
    out = ids.gather(1, torch.argmax(probs / N[:, :m], -1, keepdim=True))[:, 0]
    if top_k > m:
        for r in torch.nonzero(~(cum[:, -2] > top_p)).flatten().tolist():
            out[r] = sample_with_noise(L[r], temperature, top_k, top_p, N[r], keep_crossing=True, m=m)
    return out


def _draw_many(tower, logits, windows, pres, temperature, top_p, top_k):
    """port10 DRAMA3_CB_VDRAW=1: _draw for n streams at once (the same per-row math and noise) -> (mains, code0s,
    redrawn flags, host rows [5, n, H])."""
    from models.demos.blackhole.drama3.host.tower import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P

    t = tower.host
    top_k = top_k or len(t.candidates)
    L = torch.stack(logits)
    first = t.candidates[_sample_rows(L, temperature, top_k, top_p, torch.stack([p["n1"] for p in pres]))]
    ras = t.candidates[_sample_rows(L, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, torch.stack([p["n2"] for p in pres]))]
    first_l, ras_l = first.tolist(), ras.tolist()
    mains = [r if (f in w and t.sem0 <= f < t.sem0 + t.cb_size) else f for f, r, w in zip(first_l, ras_l, windows)]
    m = torch.tensor(mains)
    c0 = (m - t.sem0).clamp(0, t.cb_size - 1)
    lab = t.is_semantic(m)
    si = torch.cat([p["si"] for p in pres])
    rows = [tower._emb_rows(layer, c0, lab, si if i == 0 else None) for i, layer in enumerate(tower.shared)]
    rows.append(tower._emb_rows(tower.residual[0], c0, lab))
    return mains, c0.tolist(), [f != mm for f, mm in zip(first_l, mains)], torch.stack(rows)  # [5, n, H]


def _pre(tower, frame, gen):
    """prepare_frame without the device upload: the solo RNG order (main draw, RAS draw, 9 heads)."""
    t = tower.host
    n = len(t.candidates)
    n1 = torch.empty(n).exponential_(1, generator=gen)
    n2 = torch.empty(n).exponential_(1, generator=gen)
    heads = torch.stack([torch.empty(t.cb_size).exponential_(1, generator=gen) for _ in tower.residual])
    return dict(n1=n1, n2=n2, heads=heads, si=t.shared_input(frame.reshape(1, -1)))


def run_batch(s, st, jobs, max_frames, cc, temperature=1.0, top_p=0.9, top_k=None):
    """jobs: up to B (rows, seed). All start together; a short line's slot runs on until the longest ends.
    -> per job (frames, timing, audio)."""
    from models.demos.blackhole.drama3.host.stream_codec import MultiCardStreams

    B = st.B
    n = len(jobs)
    assert 1 <= n <= B
    tw, bb = s.tower, s.bb
    P = s.P
    t_start = time.time()
    timing = [dict(prompt_rows=r.shape[0], prefix_rows=P, batch=B, batch_jobs=n) for r, _ in jobs]
    codec = MultiCardStreams(cc, s.card_codec, n)
    # 1) prefill each stream with the one-row path in its page-table row
    states, normed_last, logits = [], [], []
    for b, (rows, _) in enumerate(jobs):
        t0 = time.time()
        assert bb.bucket_for(rows.shape[0] - P) == bb.bucket_for(s._warm_T - P)
        st.stream_table(b)
        if P and (s.voice is None or not torch.equal(s.voice, rows[: P + 1])):
            st.stream_table(0)  # the prefix blocks are stream 0's
            t1 = time.time()
            s.set_voice(rows)
            timing[b]["voice_prefill_s"] = time.time() - t1
            st.stream_table(b)
        normed, lg = s._prefill_line(rows)
        states.append(st.read_gdn())
        normed_last.append(normed[-1])
        logits.append(lg)
        timing[b]["prefill_s"] = time.time() - t0
    for b in range(n, B):  # empty slots repeat job 0's state (their output is dropped)
        states.append(states[0])
        normed_last.append(normed_last[0])
    st.stream_table(None)
    t0 = time.time()
    st.load_gdn(states)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.stack(normed_last).reshape(1, B, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), st.dec_out[0]
    )
    t_load = time.time() - t0
    # 2) decode all streams together
    gens = [torch.Generator().manual_seed(seed) for _, seed in jobs]
    pos = torch.tensor([r.shape[0] - 1 for r, _ in jobs] + [jobs[0][0].shape[0] - 1] * (B - n), dtype=torch.long)
    inp = [r[r.shape[0] - 1] for r, _ in jobs] + [jobs[0][0][-1]] * (B - n)
    pre = [_pre(tw, inp[b], gens[b]) for b in range(n)]
    windows = [[] for _ in range(n)]
    frames = [[] for _ in range(n)]
    done = [False] * n
    ras = [0] * n
    t_bb = t_tw = t_host = t_overlap = 0.0
    first_frame = [None] * n
    steps = 0
    last_frame = [torch.zeros(11, dtype=torch.long)] * B
    noise = torch.ones(1, len(tw.residual), B, st.K)
    for i in range(max_frames):
        t0 = time.time()
        host_rows = torch.zeros(st.n_host, B, tw.H)
        mains = [0] * B
        code0s = [0] * B
        for b in range(n):
            if done[b]:
                continue
            main, code0, redrawn, rows5 = _draw(tw, logits[b], windows[b], pre[b], temperature, top_p, top_k)
            mains[b], code0s[b] = main, code0
            ras[b] += int(redrawn)
            host_rows[:, b] = rows5
            noise[0, :, b] = pre[b]["heads"][:, : st.K]
        if i == 0:  # later frames: the backbone decode staged these positions already (shared buffers)
            st.stage_pos(pos)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), st.noise_buf)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host_rows.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), st.e_host_buf
        )
        t_host += time.time() - t0
        t0 = time.time()
        ttnn.execute_trace(st.mesh, st.tw_tid, cq_id=0, blocking=False)
        codes = ttnn.to_torch(st.codes_out).float().reshape(B, -1)[:, : len(tw.residual)].round().long()
        t_tw += time.time() - t0
        steps += 1
        t0 = time.time()
        for b in range(n):
            if done[b]:
                continue
            if first_frame[b] is None:
                first_frame[b] = time.time() - t_start
            if mains[b] == tw.host.eos:
                done[b] = True
                codec.end(b)
                timing[b]["gen_s"] = time.time() - t_start
                continue
            frame = torch.cat([torch.tensor([mains[b], code0s[b]]), codes[b]])
            frames[b].append(frame)
            codec.push(b, frame[1:])
            windows[b] = (windows[b] + [mains[b]])[-16:]
            last_frame[b] = frame
        codec.poll()
        if all(done):
            t_host += time.time() - t0
            break
        pos = pos + 1
        emb = s.embedder(torch.stack(last_frame))
        t_host += time.time() - t0
        t0 = time.time()
        st.stage(emb, pos)
        ttnn.execute_trace(st.mesh, st.bb_tid, cq_id=0, blocking=False)
        # the next frame's logit-independent host draws run while the backbone trace executes (solo: HOST_OVERLAP)
        t1 = time.time()
        for b in range(n):
            if not done[b]:
                pre[b] = _pre(tw, last_frame[b], gens[b])
        t_overlap += time.time() - t1
        lg_all = ttnn.to_torch(st.dec_out[1]).float().reshape(B, -1)[:, : bb.n_cand]
        logits = [lg_all[b] for b in range(n)]
        t_bb += time.time() - t0
        if (i + 1) % 50 == 0:
            log(f"  step {i + 1}: backbone {t_bb / steps * 1e3:.1f} ms, tower {t_tw / steps * 1e3:.1f} ms, host {t_host / steps * 1e3:.1f} ms per step ({n} streams)")
    for b in range(n):
        if not done[b]:
            codec.end(b)
            timing[b]["gen_s"] = time.time() - t_start
    t_gen_end = time.time()
    audios = codec.finish()
    out = []
    for b in range(n):
        f = torch.stack(frames[b]) if frames[b] else torch.zeros(0, 11, dtype=torch.long)
        t = timing[b]
        t.update(
            frames=f.shape[0],
            steps=steps,
            gdn_load_s=t_load,
            decode_backbone_s=t_bb,
            tower_steps_s=t_tw,
            host_s=t_host,
            backbone_ms_per_step=1e3 * t_bb / max(steps, 1),
            tower_ms_per_step=1e3 * t_tw / max(steps, 1),
            host_ms_per_step=1e3 * t_host / max(steps, 1),
            overlap_host_ms_per_step=1e3 * t_overlap / max(steps, 1),
            ras_redraws=ras[b],
            first_frame_s=first_frame[b],
            stopped_by_eos=done[b],
            ttfa_s=(codec.first_audio_t[b] - t_start) if codec.first_audio_t[b] else None,
            audio_done_s=(codec.done_t[b] - t_start) if codec.done_t[b] else None,
            card_codec_s=codec.card_s[b],
            codec_tail_s=time.time() - t_gen_end,
            batch_wall_s=time.time() - t_start,
        )
        out.append((f, t, audios[b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="all")
    ap.add_argument("--lines-file", default="", help="JSON list of {id, spoken, written}: text = spoken")
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="comma list, one per job slot of the first batch (B copies of one line)")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--order", default="length", help="length: batches of similar length (longest first); given: as listed")
    ap.add_argument("--codec-threads", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["DRAMA3_KV_STREAMS"] = str(a.batch_size)
    os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"
    import yaml

    from models.demos.blackhole.drama3.host.codec import save_wav
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.drama3.host.stream_codec import CodecProcess
    from models.demos.blackhole.drama3.tt.device_check import dram_used_gb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.lines_file:
        lines = {l["id"]: dict(l, text=l["spoken"]) for l in json.load(open(a.lines_file))}
    else:
        lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ids = list(lines) if a.lines == "all" else a.lines.split(",")
    ref_text = Path(a.ref).with_suffix(".txt").read_text().strip()
    codes = reference_codes(a.ref, "/opt/ep0/day3")
    t0 = time.time()
    cc = CodecProcess(threads=a.codec_threads, compile=False, multi=True)
    log(f"codec (multi-stream latents) ready in {time.time() - t0:.0f}s")
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=int(os.environ.get("DRAMA3_L1_SMALL", 65536)), trace_region_size=256 << 20)
    try:
        s = FastSession(mesh)
        prompts = {lid: s.prompt(lines[lid]["text"], [(ref_text, codes)]) for lid in ids}
        s._warm_T = max(r.shape[0] for r in prompts.values())
        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + list(prompts.values()))
        s.warmup(prompts[ids[0]], capture=False)
        st = BatchState(s, a.batch_size)
        t0 = time.time()
        st.capture()
        if s.card_codec is not None:
            s.card_codec.capture()
        log(f"batch {a.batch_size}: traces captured in {time.time() - t0:.1f}s, card DRAM {dram_used_gb(mesh)[0]:.2f} GB")
        timing["_batch"] = dict(batch_size=a.batch_size, dram_gb=round(dram_used_gb(mesh)[0], 2))
        if a.seeds:  # B copies of the first line, one seed per slot (the per-row identity check vs solo runs)
            seeds = [int(x) for x in a.seeds.split(",")]
            batches = [[(ids[0], sd, f"{a.tag}_s{sd}") for sd in seeds]]
        else:
            order = sorted(ids, key=lambda i: -prompts[i].shape[0]) if a.order == "length" else ids
            jobs = [(lid, a.seed, a.tag) for lid in order]
            batches = [jobs[i : i + a.batch_size] for i in range(0, len(jobs), a.batch_size)]
        for bi, batch in enumerate(batches):
            t_b = time.time()
            res = run_batch(s, st, [(prompts[lid], sd) for lid, sd, _ in batch], a.max_frames, cc)
            wall = time.time() - t_b
            audio_tot = 0.0
            for (lid, sd, tag), (frames, t, audio) in zip(batch, res):
                t0 = time.time()
                save_wav(out / f"{lid}{tag}.wav", audio, cc.sr)
                t["save_s"] = round(time.time() - t0, 3)
                t["audio_s"] = round(frames.shape[0] * cc.frame_length / cc.sr, 3)
                audio_tot += t["audio_s"]
                # compute_s: batch start .. this stream's audio complete; rtf per stream on that
                t["compute_s"] = t["audio_done_s"]
                t["rtf"] = round(t["compute_s"] / t["audio_s"], 3) if t["audio_s"] else None
                t["batch_index"] = bi
                t["start_epoch"], t["end_epoch"] = t_b, t_b + t["batch_wall_s"]
                t = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}
                torch.save(frames, out / f"{lid}{tag}.frames.pt")
                timing[lid + tag] = t
                log(f"{lid}{tag}: {json.dumps(t)}")
            timing[f"_batch{bi}"] = dict(
                jobs=[lid + tag for lid, _, tag in batch], wall_s=round(wall, 3), audio_s=round(audio_tot, 3),
                audio_per_wall=round(audio_tot / wall, 3),
            )
            log(f"batch {bi}: {len(batch)} streams, {audio_tot:.1f} s audio in {wall:.1f} s wall = {audio_tot / wall:.2f} audio s per s")
            json.dump(timing, open(timing_path, "w"), indent=1)
    finally:
        ttnn.close_device(mesh)
        cc.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
