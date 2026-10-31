import os

os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/drama3")


def edit(p, pairs):
    raw = open(p, "rb").read()
    crlf = b"\r\n" in raw
    s = raw.decode().replace("\r\n", "\n")
    for o, n in pairs:
        assert s.count(o) == 1, (p, o)
        s = s.replace(o, n)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode())


MULTI_WORKER = '''

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
'''

MULTI_STREAMS = '''

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
'''

edit(
    "host/stream_codec.py",
    [
        (
            "\n\nclass CodecProcess:\n",
            MULTI_WORKER + "\n\nclass CodecProcess:\n",
        ),
        (
            "    def __init__(self, threads=6, compile=True, tail_threads=None):\n"
            "        import multiprocessing as mp\n"
            "\n"
            "        ctx = mp.get_context(\"spawn\")\n"
            "        self.inq, self.outq = ctx.Queue(), ctx.Queue()\n"
            "        self.p = ctx.Process(target=_worker, args=(self.inq, self.outq, threads, compile, tail_threads), daemon=True)\n",
            "    def __init__(self, threads=6, compile=True, tail_threads=None, multi=False):\n"
            "        import multiprocessing as mp\n"
            "\n"
            "        ctx = mp.get_context(\"spawn\")\n"
            "        self.inq, self.outq = ctx.Queue(), ctx.Queue()\n"
            "        if multi:  # batch mode: several latent streams at once (MultiCardStreams)\n"
            "            self.p = ctx.Process(target=_multi_worker, args=(self.inq, self.outq, threads), daemon=True)\n"
            "        else:\n"
            "            self.p = ctx.Process(target=_worker, args=(self.inq, self.outq, threads, compile, tail_threads), daemon=True)\n",
        ),
    ],
)
s = open("host/stream_codec.py", "rb").read().decode()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.rstrip() + nl + MULTI_STREAMS.replace("\n", nl)
open("host/stream_codec.py", "wb").write(s.encode())

edit(
    "tt/tower_device.py",
    [
        (
            "        if self.row_norm is not None and x.shape[-2] == 1:\n",
            "        if self.row_norm is not None and x.shape[-2] <= decode_norm.ROWS:  # ROWS > 1 only in batch mode\n",
        ),
    ],
)
s = open("tt/tower_device.py", "rb").read().decode()
nl = "\r\n" if "\r\n" in s else "\n"
anchor = "from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention"
assert anchor in s
s = s.replace(anchor, "from models.demos.blackhole.drama3.tt import decode_norm" + nl + anchor, 1)
open("tt/tower_device.py", "wb").write(s.encode())
print("PATCHED")
