#!/usr/bin/env python3
"""Prefill speed of a running server: one request at a time, no shared prefix between sizes.

    prefill.py <label> [--url http://127.0.0.1:8087] [--sizes 1024,4096,8192,32768] [--repeats 3] [--out file.json]

Each request: a fresh slice of source code (the `check/prefill_text.txt` next to this script if it exists, else
every *.py under --src), greedy, thinking off, 4 tokens out. TTFT = first streamed token minus send; prefill tok/s =
prompt tokens / TTFT (includes the fixed per-request cost, so it is a floor for the pure prefill rate). The first
request of each size is a warm-up and is not counted. Standard library only.
"""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default="http://127.0.0.1:8087")
ap.add_argument("--sizes", default="1024,4096,8192,32768")
ap.add_argument("--repeats", type=int, default=3)
_here = pathlib.Path(__file__).resolve().parent
ap.add_argument("--src", default=str(_here.parent / "qwen36" if (_here.parent / "qwen36").exists() else _here / "qwen36"))
ap.add_argument("--out", default="")
a = ap.parse_args()


def post(path, body):
    req = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


txt = pathlib.Path(__file__).resolve().with_name("prefill_text.txt")
if txt.exists():
    text = txt.read_text(errors="ignore")
else:
    text = "\n".join(p.read_text(errors="ignore") for p in sorted(pathlib.Path(a.src).rglob("*.py")))
toks = json.load(post("/tokenize", {"model": "qwen38", "prompt": text[:4_000_000], "add_special_tokens": False}))["tokens"]
if len(toks) < 2 * max(int(x) for x in a.sizes.split(",")):
    toks = toks * 4
print(f"corpus: {len(toks)} tokens", flush=True)
pos = 0


def one(n):
    global pos
    if pos + n > len(toks):
        pos = 0
    chunk = toks[pos:pos + n - 64]  # room for the chat template
    pos += n
    prompt = json.load(post("/detokenize", {"model": "qwen38", "tokens": chunk}))["prompt"]
    body = {"model": "qwen38", "messages": [{"role": "user", "content": "Summarize this code in one sentence.\n\n" + prompt}],
            "max_tokens": 4, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    first = None
    usage = None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content") and first is None:
                    first = time.time()
    return usage["prompt_tokens"], first - t0


res = []
for n in [int(x) for x in a.sizes.split(",")]:
    one(n)
    rows = [one(n) for _ in range(a.repeats)]
    pt = rows[0][0]
    tt = statistics.median(x[1] for x in rows)
    tps = pt / tt
    res.append({"size": n, "prompt_tokens": pt, "ttft_s": round(tt, 3), "prefill_tok_s_floor": round(tps), "all": [round(x[1], 3) for x in rows]})
    print(f"PREFILL {a.label} n={n} prompt={pt} ttft_median={tt:.3f}s floor_tok_s={tps:.0f} all={[round(x[1],3) for x in rows]}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
