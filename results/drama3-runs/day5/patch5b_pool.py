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


edit(
    "tt/backbone_paged.py",
    [
        (
            "        shape = [n_blocks, args.n_kv_heads, BLOCK, args.head_dim]\n",
            "        # DRAMA3_KV_STREAMS (batch mode, default 1): a K/V pool of that many sequences' blocks; the identity page\n"
            "        # table above is stream 0's (host/batch.py maps the other streams' rows)\n"
            "        shape = [n_blocks * int(os.environ.get(\"DRAMA3_KV_STREAMS\", \"1\")), args.n_kv_heads, BLOCK, args.head_dim]\n",
        ),
    ],
)
edit(
    "tt/tower_device.py",
    [
        (
            "        shape = [max_seq // BLOCK, nkv, BLOCK, hd]\n",
            "        shape = [max_seq // BLOCK * int(os.environ.get(\"DRAMA3_KV_STREAMS\", \"1\")), nkv, BLOCK, hd]  # batch mode: a pool\n",
        ),
    ],
)
s = open("tt/tower_device.py").read()
if "\nimport os" not in s:
    nl = "\r\n" if "\r\n" in s else "\n"
    i = s.index("import torch")
    s = s[:i] + "import os" + nl + nl + s[i:]
    open("tt/tower_device.py", "w", newline="").write(s)
edit(
    "tt/device_sampler.py",
    [
        (
            "        m = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, 1, 1])  # the max\n",
            "        m = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, vals.shape[2], 1])  # the max (per row: batch mode has B rows)\n",
        ),
        (
            '        """logits [1,1,1,n] fp32 (device), noise [1,1,1,K] fp32 (device) -> (code [1,1,1,1] fp32, one-hot [1,1,1,n] bf16)."""\n',
            '        """logits [1,1,1,n] fp32 (device), noise [1,1,1,K] fp32 (device) -> (code [1,1,1,1] fp32, one-hot [1,1,1,n] bf16).\n'
            '        Batch mode: B rows on dim 2 ([1,1,B,n], [1,1,B,K] -> [1,1,B,1], [1,1,B,n])."""\n',
        ),
    ],
)
edit(
    "host/generate_fast.py",
    [
        (
            "    def warmup(self, rows):\n"
            '        """Compile every program on one sequence (prefill buckets, tower prefill, tower step), then capture."""\n',
            "    def warmup(self, rows, capture=True):\n"
            '        """Compile every program on one sequence (prefill buckets, tower prefill, tower step), then capture.\n'
            '        capture=False (batch mode, host/batch.py): no one-row traces; the batch captures its own."""\n',
        ),
        (
            "        if self.device_sampling:\n"
            "            self.tower.prepare_fused()  # its persistent buffers must exist before the backbone capture\n"
            "        self.bb.capture(pos=T)\n",
            "        if self.device_sampling:\n"
            "            self.tower.prepare_fused()  # its persistent buffers must exist before the backbone capture\n"
            "        if not capture:\n"
            "            self.voice = None\n"
            "            return\n"
            "        self.bb.capture(pos=T)\n",
        ),
    ],
)
print("PATCHED")
