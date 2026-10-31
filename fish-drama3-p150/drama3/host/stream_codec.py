# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Streaming codec decode in a background thread, with Fish's stream windows (SGLang-Omni defaults).

Windows end at frames 10, 20, 32, 64 and then every 64 more; each decodes from 32 frames before the audio
already emitted (stream_overlap_tokens) and keeps only the new samples (no crossfade, Fish's default). The
codec's conv decoder is torch.compile'd (inductor, cached under /opt/ep0/inductor_cache): same output as eager
(max |diff| ~1e-6), about 1.6x faster on this CPU.
"""

import os
import queue
import threading
import time

import torch

from models.demos.blackhole.drama3.host.codec import load_codec

STREAM_STRIDE, RAMP, FOLLOWUP, OVERLAP = 10, (10, 12, 32), 64, 32


def first_window():
    """DRAMA3_FIRST_WINDOW=k (opt-in, k < 10): one extra window ending at frame k before Fish's windows (10, 20,
    32, 64, ...), which stay as they are. Default 10 = Fish's windows unchanged."""
    return int(os.environ.get("DRAMA3_FIRST_WINDOW", STREAM_STRIDE))


def window_ends():
    ends = os.environ.get("DRAMA3_STREAM_ENDS")
    if ends:  # port10 (throughput mode, opt-in): these first window ends, then FOLLOWUP (64) frames per window
        end = 0
        for e in ends.split(","):
            end = int(e)
            yield end
        while True:
            end += FOLLOWUP
            yield end
    ramp = os.environ.get("DRAMA3_STREAM_RAMP")
    if ramp:  # port6: windows sized for gapless playback at a generation RTF up to r (see ramp_ends)
        r, _, pre = ramp.partition(":")
        yield from ramp_ends(float(r), prebuffer_s=float(pre or 0))
        return
    end = 0
    if first_window() < STREAM_STRIDE:
        yield first_window()
    for s in (STREAM_STRIDE, *RAMP):
        end += s
        yield end
    while True:
        end += FOLLOWUP
        yield end


FRAME_S = 2048 / 44100  # audio seconds per codec frame


def ramp_ends(r, w_max=64, prebuffer_s=0.0):
    """DRAMA3_STREAM_RAMP=r[:prebuffer_s] (port6, opt-in): window ends for gapless playback when frames come at
    r x real time and playback starts prebuffer_s after the first window (w1 = FIRST_WINDOW frames). The window
    over frames [a, a + w) is ready (a + w) * r frame-times after the start and needed w1 * r + p + a after it
    (p = the pre-buffer in frames), so w <= (a * (1 - r) + w1 * r + p) / r, capped at w_max (the card decoder
    holds 64 new + 32 overlap frames). Fish's schedule (10, 20, 32, 64, 128, ...) needs r <= 0.5 at its
    64 -> 128 step."""
    w1 = first_window()
    p = prebuffer_s / FRAME_S
    end = w1
    yield end
    while True:
        w = int(min(w_max, max(w1, (end * (1.0 - r) + w1 * r + p) / r)))
        end += w
        yield end


def playback_check(chunk_t, sr):
    """chunk_t: [(arrival time, samples)] of one stream. Playback starts at the first chunk; chunk i (samples
    [c_(i-1), c_i)) must arrive before playback reaches c_(i-1). -> (underruns, worst lateness s, least lead s).
    The worst lateness is the playback pre-buffer that would have made this stream gapless."""
    if not chunk_t:
        return None, None, None
    t_first = chunk_t[0][0]
    played, under, worst, ahead = 0, 0, 0.0, None
    for t, n in chunk_t:
        late = t - (t_first + played / sr)
        if late > 1e-3:
            under += 1
            worst = max(worst, late)
        elif played:
            ahead = -late if ahead is None else min(ahead, -late)
        played += n
    return under, round(worst, 3), (round(ahead, 3) if ahead is not None else None)


class CompiledCodec:
    def __init__(self, compile=True, threads=8):
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/opt/ep0/inductor_cache")
        self.codec = load_codec()
        self.codec.fold_weight_norm()  # as Fish's vocoder stage does; numerically identical
        if os.environ.get("DRAMA3_QUANT_FAST", "0") == "1":  # port6 (opt-in): faster CPU op forms, host/quant_fast.py
            from models.demos.blackhole.drama3.host import quant_fast

            quant_fast.install(self.codec.quantizer)
        self.sr, self.frame_length = self.codec.sample_rate, self.codec.frame_length
        self.decoder = torch.compile(self.codec.decoder, dynamic=True) if compile else self.codec.decoder
        self.threads = threads

    @torch.inference_mode()
    def decode(self, codes):
        """codes [10, n] -> waveform [n * frame_length]."""
        return self.decoder(self.codec.quantizer.decode(codes[None].long()))[0, 0]

    def warmup(self):
        for n in (10, 20, 32, 64, 96):
            self.decode(torch.zeros(10, n, dtype=torch.long))


class StreamDecoder:
    """push(frame codes [10]) from the generation loop; the worker decodes each window as soon as it is full."""

    def __init__(self, cc: CompiledCodec):
        self.cc = cc
        self.q = queue.Queue()
        self.codes, self.chunks = [], []
        self.emitted = 0
        self.first_audio_t = None
        self.decode_s = 0.0
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def push(self, frame_codes):
        self.q.put(frame_codes)

    def finish(self):
        self.q.put(None)
        self.t.join()
        if self.error:
            raise self.error
        return torch.cat(self.chunks) if self.chunks else torch.zeros(0)

    def _window(self, end):
        start = max(0, self.emitted - OVERLAP)
        t0 = time.time()
        audio = self.cc.decode(torch.stack(self.codes[start:end], 1))
        self.decode_s += time.time() - t0
        self.chunks.append(audio[(self.emitted - start) * self.cc.frame_length :].float())
        self.emitted = end
        if self.first_audio_t is None:
            self.first_audio_t = time.time()

    def _run(self):
        torch.set_num_threads(self.cc.threads)
        ends = window_ends()
        nxt = next(ends)
        try:
            while True:
                item = self.q.get()
                if item is None:
                    if len(self.codes) > self.emitted:
                        self._window(len(self.codes))
                    return
                self.codes.append(item)
                while len(self.codes) >= nxt:
                    self._window(nxt)
                    nxt = next(ends)
        except Exception as e:  # surfaced by finish()
            self.error = e


def _worker(inq, outq, threads, compile, tail_threads=None):
    """Codec worker process: one stream per ("start",) ... frames ... ("end",). tail_threads: the windows left
    at ("end",) run with this many threads (generation is over, its host cores are free)."""
    torch.set_num_threads(threads)
    cc = CompiledCodec(compile=compile, threads=threads)
    cc.warmup()
    outq.put(("ready", cc.sr, cc.frame_length))
    while True:
        msg = inq.get()
        if msg[0] == "quit":
            return
        assert msg[0] in ("start", "start_latent")
        latent = msg[0] == "start_latent"  # card codec: send the window's quantizer latent, the card decodes it
        codes, emitted, decode_s, first = [], 0, 0.0, None
        ends = window_ends()
        nxt = next(ends)

        def window(end):
            nonlocal emitted, decode_s, first
            start = max(0, emitted - OVERLAP)
            t0 = time.time()
            if latent:
                with torch.inference_mode():
                    z = cc.codec.quantizer.decode(torch.stack(codes[start:end], 1)[None].long())  # [1, 1024, 4n]
                decode_s += time.time() - t0
                outq.put(("z", z[0].T.contiguous().numpy(), emitted - start, end - start))
            else:
                audio = cc.decode(torch.stack(codes[start:end], 1))
                decode_s += time.time() - t0
                outq.put(("chunk", audio[(emitted - start) * cc.frame_length :].float().numpy()))
            emitted = end
            if first is None:
                first = time.time()

        while True:
            m = inq.get()
            if m[0] == "end":
                if tail_threads:
                    torch.set_num_threads(tail_threads)
                if len(codes) > emitted:
                    window(len(codes))
                torch.set_num_threads(threads)
                outq.put(("done", decode_s, first))
                break
            codes.append(torch.from_numpy(m[1]))
            while len(codes) >= nxt:
                window(nxt)
                nxt = next(ends)


def _multi_worker(inq, outq, threads):
    """Batch mode (host/batch.py): the window latents of several streams at once, messages tagged with a stream
    id: ("start", sid), ("frame", sid, codes), ("end", sid) -> ("z", sid, z, keep_from, n), ("done", sid, decode_s).
    Same windows as _worker's latent mode, per stream."""
    torch.set_num_threads(threads)
    cc = CompiledCodec(compile=False, threads=threads)
    outq.put(("ready", cc.sr, cc.frame_length))
    st = {}

    def window(sid, end):
        s = st[sid]
        start = max(0, s["emitted"] - OVERLAP)
        t0 = time.time()
        with torch.inference_mode():
            z = cc.codec.quantizer.decode(torch.stack(s["codes"][start:end], 1)[None].long())
        s["decode_s"] += time.time() - t0
        outq.put(("z", sid, z[0].T.contiguous().numpy(), s["emitted"] - start, end - start))
        s["emitted"] = end

    while True:
        m = inq.get()
        if m[0] == "quit":
            return
        sid = m[1]
        if m[0] == "start":
            ends = window_ends()
            st[sid] = dict(codes=[], emitted=0, decode_s=0.0, ends=ends, nxt=next(ends))
        elif m[0] == "frame":
            s = st[sid]
            s["codes"].append(torch.from_numpy(m[2]))
            while len(s["codes"]) >= s["nxt"]:
                window(sid, s["nxt"])
                s["nxt"] = next(s["ends"])
        elif m[0] == "end":
            s = st[sid]
            if len(s["codes"]) > s["emitted"]:
                window(sid, len(s["codes"]))
            outq.put(("done", sid, s["decode_s"]))
            del st[sid]


class CodecProcess:
    """The streaming codec in its own process (no GIL / core contention with the generation loop's host work).
    Same windows and output as StreamDecoder."""

    def __init__(self, threads=6, compile=True, tail_threads=None, multi=False):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        if multi:  # batch mode: several latent streams at once (MultiCardStreams)
            self.p = ctx.Process(target=_multi_worker, args=(self.inq, self.outq, threads), daemon=True)
        else:
            self.p = ctx.Process(target=_worker, args=(self.inq, self.outq, threads, compile, tail_threads), daemon=True)
        self.p.start()
        tag, self.sr, self.frame_length = self.outq.get(timeout=900)
        assert tag == "ready"

    def stream(self, card=None):
        """card: a CardCodec -> the process sends quantizer latents and the card decodes them (poll() each frame).
        A CardCodec with the quantizer on the card too (DRAMA3_QUANT_DEVICE=1) -> DeviceQuantStream (no process)."""
        if card is not None and getattr(card, "dq", None) is not None:
            return DeviceQuantStream(card)
        return _ProcStream(self) if card is None else _CardStream(self, card)

    def close(self):
        self.inq.put(("quit",))
        self.p.join(timeout=30)


class _ProcStream:
    def __init__(self, cp):
        self.cp = cp
        cp.inq.put(("start",))
        self.chunks = []
        self.decode_s, self.first_audio_t = 0.0, None

    def push(self, frame_codes):
        self.cp.inq.put(("frame", frame_codes.numpy()))

    def finish(self):
        self.cp.inq.put(("end",))
        while True:
            m = self.cp.outq.get(timeout=600)
            if m[0] == "chunk":
                self.chunks.append(torch.from_numpy(m[1]))
            else:
                _, self.decode_s, self.first_audio_t = m
                break
        return torch.cat(self.chunks) if self.chunks else torch.zeros(0)


class CardCodec:
    """The codec decoder on the card (tt/codec_device.py), fed with the codec process's window latents. A window
    is decoded at the smallest fixed length in WINDOWS that holds it (zero latent rows appended): the decoder is
    causal, so the padding does not touch the real samples, and each length compiles once in warm-up.
    DRAMA3_CODEC_WINDOWS (default "32,64,96"; "96" = one size): needs l1_small_size 65536 (generate_fast default;
    with the GDN minimum 24576 the second decoder runs out of L1_SMALL: conv1d keeps
    per-program sliding-window buffers there; the GDN conv needs l1_small_size 24576) at the second decoder."""

    def __init__(self, mesh):
        from models.demos.blackhole.drama3.tt.codec_device import DeviceDecoder

        codec = load_codec()
        codec.fold_weight_norm()
        self.frame_length, self.hop = codec.frame_length, codec.frame_length // 4
        self.windows = sorted(int(w) for w in os.environ.get("DRAMA3_CODEC_WINDOWS", "32,64,96").split(","))
        assert self.windows[-1] >= 96, "the longest Fish window is 64 new + 32 overlap frames"
        self.decs = {}
        for w in self.windows:  # compile + weight preparation (before any trace capture)
            self.decs[w] = DeviceDecoder(mesh, codec, 4 * w)
            self.decs[w](torch.zeros(1, 4 * w, 1024))
        self.dec = self.decs[self.windows[-1]]

        # port6 DRAMA3_QUANT_DEVICE=1: the quantizer decode on the card as well (tt/quant_device.py)
        self.dq = None
        if os.environ.get("DRAMA3_QUANT_DEVICE", "0") == "1":
            from models.demos.blackhole.drama3.tt.quant_device import DeviceQuantizer

            self.dq = DeviceQuantizer(mesh, codec, sizes=tuple(self.windows))
            for w in self.windows:  # the decoders on a bf16 card latent are other programs: compile them now
                self.decode_dev(torch.zeros(10, w, dtype=torch.long), 0, w)

    def decode_dev(self, codes, keep_from, n):
        """Window codes [10, n] -> samples of frames keep_from..n-1, quantizer and decoder on the card."""
        import ttnn

        w = next(w for w in self.windows if w >= n)
        z = self.dq(codes, w)
        dt = getattr(self, "dev_traces", {}).get(w)
        if dt is not None:  # port10 DRAMA3_CODEC_DEV_TRACE=1: the decoder replayed on the quantizer trace's output
            tid, out, L = dt
            ttnn.execute_trace(self.dq.mesh, tid, cq_id=0, blocking=False)
            audio = self.decs[w].audio(out)
            return audio[keep_from * self.frame_length : n * self.frame_length]
        audio = self.decs[w](z)
        if w not in self.dq.traces:
            ttnn.deallocate(z)
        return audio[keep_from * self.frame_length : n * self.frame_length]

    def capture(self):
        """DRAMA3_CODEC_TRACE=1: replay the decoders as traces (call after the generation traces are captured)."""
        if os.environ.get("DRAMA3_CODEC_TRACE", "0") == "1":
            for d in self.decs.values():
                d.capture()
        if self.dq is not None:
            self.dq.capture()
            if os.environ.get("DRAMA3_CODEC_DEV_TRACE", "0") == "1":
                # port10: each decoder as a trace whose input is the quantizer trace's persistent output z (same
                # programs as the eager decode_dev path, without the per-op host dispatch)
                import ttnn

                self.dev_traces = {}
                for w, d in self.decs.items():
                    if w not in self.dq.traces:
                        continue
                    z = self.dq.traces[w][2]
                    x, _ = d._body(z)
                    ttnn.deallocate(x)
                    ttnn.synchronize_device(self.dq.mesh)
                    tid = ttnn.begin_trace_capture(self.dq.mesh, cq_id=0)
                    out, _ = d._body(z)
                    ttnn.end_trace_capture(self.dq.mesh, tid, cq_id=0)
                    ttnn.synchronize_device(self.dq.mesh)
                    self.dev_traces[w] = (tid, out, d.L_out)

    def decode(self, z_rows, keep_from, n):
        """z_rows [4n, 1024] -> samples of frames keep_from..n-1."""
        w = next(w for w in self.windows if w >= n)
        z = torch.zeros(1, 4 * w, z_rows.shape[-1])
        z[0, : z_rows.shape[0]] = torch.from_numpy(z_rows)
        audio = self.decs[w](z)
        return audio[keep_from * self.frame_length : n * self.frame_length]


class _CardStream:
    def __init__(self, cp, card):
        self.cp, self.card = cp, card
        cp.inq.put(("start_latent",))
        self.chunks = []
        self.decode_s, self.card_s, self.first_audio_t = 0.0, 0.0, None
        self.first_z_t = self.first_card_s = None  # TTFA breakdown: first latent taken, its card decode
        self.chunk_t = []  # port6: (time each chunk is decoded, its samples), for playback_check

    def push(self, frame_codes):
        self.cp.inq.put(("frame", frame_codes.numpy()))

    def _take(self, m):
        _, z, keep_from, n = m
        t0 = time.time()
        self.chunks.append(self.card.decode(z, keep_from, n))
        self.card_s += time.time() - t0
        self.chunk_t.append((time.time(), self.chunks[-1].shape[0]))
        if self.first_audio_t is None:
            self.first_audio_t = time.time()
            self.first_z_t, self.first_card_s = t0, self.first_audio_t - t0

    def wait_first(self):
        """DRAMA3_FIRST_WINDOW_WAIT=1: block until the first window's latent arrives and decode it now (instead of
        at the poll after the next frame)."""
        if self.first_audio_t is None:
            self._take(self.cp.outq.get(timeout=60))

    def poll(self):
        """Decode every latent window that has arrived (called by the generation loop between frames)."""
        while True:
            try:
                m = self.cp.outq.get_nowait()
            except queue.Empty:
                return
            self._take(m)

    def finish(self):
        self.cp.inq.put(("end",))
        while True:
            m = self.cp.outq.get(timeout=600)
            if m[0] == "z":
                self._take(m)
            else:
                _, self.decode_s, _ = m
                break
        return torch.cat(self.chunks) if self.chunks else torch.zeros(0)


class DeviceQuantStream:
    """port6 (DRAMA3_QUANT_DEVICE=1): Fish's stream windows (window_ends, 32-frame overlap) with the quantizer AND the
    decoder on the card, in the generation loop itself: a window is decoded as soon as its last frame is pushed.
    Same interface as _CardStream."""

    def __init__(self, card):
        self.card = card
        self.codes, self.chunks, self.chunk_t = [], [], []
        self.emitted = 0
        self.ends = window_ends()
        self.nxt = next(self.ends)
        self.decode_s, self.card_s, self.first_audio_t = 0.0, 0.0, None
        self.first_z_t = self.first_card_s = None

    def _window(self, end):
        start = max(0, self.emitted - OVERLAP)
        t0 = time.time()
        a = self.card.decode_dev(torch.stack(self.codes[start:end], 1), self.emitted - start, end - start)
        t1 = time.time()
        self.card_s += t1 - t0
        self.chunks.append(a)
        self.chunk_t.append((t1, a.shape[0]))
        self.emitted = end
        if self.first_audio_t is None:
            self.first_audio_t = t1
            self.first_z_t, self.first_card_s = t0, t1 - t0

    def push(self, frame_codes):
        self.codes.append(frame_codes.long())
        while len(self.codes) >= self.nxt:
            self._window(self.nxt)
            self.nxt = next(self.ends)

    def wait_first(self):
        pass

    def poll(self):
        pass

    def finish(self):
        if len(self.codes) > self.emitted:
            self._window(len(self.codes))
        return torch.cat(self.chunks) if self.chunks else torch.zeros(0)


class MultiCardStreams:
    """Batch mode: n streams through one multi-stream codec process (CodecProcess(multi=True)); every arriving
    window latent is decoded on the card (CardCodec) between frames, per stream."""

    def __init__(self, cp, card, n):
        self.cp, self.card, self.n = cp, card, n
        self.chunks = [[] for _ in range(n)]
        self.card_s = [0.0] * n
        self.decode_s = [0.0] * n
        self.first_audio_t = [None] * n
        self.done_t = [None] * n
        self.ended = [False] * n
        for b in range(n):
            cp.inq.put(("start", b))

    def push(self, b, frame_codes):
        self.cp.inq.put(("frame", b, frame_codes.numpy()))

    def end(self, b):
        if not self.ended[b]:
            self.ended[b] = True
            self.cp.inq.put(("end", b))

    def _take(self, m):
        if m[0] == "z":
            _, b, z, keep_from, n = m
            t0 = time.time()
            self.chunks[b].append(self.card.decode(z, keep_from, n))
            self.card_s[b] += time.time() - t0
            if self.first_audio_t[b] is None:
                self.first_audio_t[b] = time.time()
        else:
            _, b, self.decode_s[b] = m
            self.done_t[b] = time.time()

    def poll(self):
        while True:
            try:
                m = self.cp.outq.get_nowait()
            except queue.Empty:
                return
            self._take(m)

    def finish(self):
        for b in range(self.n):
            self.end(b)
        while not all(t is not None for t in self.done_t):
            self._take(self.cp.outq.get(timeout=600))
        return [torch.cat(c) if c else torch.zeros(0) for c in self.chunks]
