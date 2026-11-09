#!/usr/bin/env python3
"""Decode speed and time to first token on the 8 code prompts in prompts.json, one request at a time.

    speed.py run TAG [--url http://127.0.0.1:8087] [--rounds 2] [--max-tokens 256]
    speed.py cmp TAG_A TAG_B    per prompt: same text?, acceptance, tok/s

Writes <dir>/speed-TAG.json (env CHECK_DIR, default ./runs). Reported separately: TTFT (median, first streamed token
minus send) and decode tok/s (mean over requests; (tokens - 1) / (time after the first token)). One warm-up request
first, not counted. Standard library only.
"""
import argparse
import json
import os
import pathlib
import statistics as st
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
W = os.environ.get("CHECK_DIR", "runs")


def req(url, prompt, mt):
    body = {"model": "qwen38", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    r_ = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    tl = None
    n = None
    txt = []
    with urllib.request.urlopen(r_, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage") and d["usage"].get("completion_tokens"):
                n = int(d["usage"]["completion_tokens"])
            c = (d.get("choices") or [{}])[0].get("delta", {}) if d.get("choices") else {}
            s = c.get("content") or ""
            if s:
                now = time.time()
                ttft = ttft if ttft is not None else now - t0
                tl = now
                txt.append(s)
    return dict(text="".join(txt), n=n, ttft=ttft, tps=(n - 1) / (tl - t0 - ttft) if n and n > 1 else None)


def stats(url):
    with urllib.request.urlopen(url + "/dflash/stats", timeout=30) as r:
        return json.load(r)


def summ(tag, rs):
    a = [r["accept_mean"] for r in rs if r.get("accept_mean") is not None]
    t = [r["tps"] for r in rs if r.get("tps")]
    tt = [r["ttft"] for r in rs if r.get("ttft")]
    tps = [r["tokens_per_step"] for r in rs if r.get("tokens_per_step")]
    print(f"SPEED {tag}: {len(rs)} requests, decode tok/s mean {st.mean(t):.2f} (min {min(t):.1f} max {max(t):.1f}), "
          f"TTFT median {st.median(tt):.3f} s, accepted drafts per step mean {st.mean(a) if a else 0:.3f}, tokens per step {st.mean(tps) if tps else 0:.3f}")


def run(a):
    rows = json.load(open(HERE / "prompts.json", encoding="utf-8"))
    req(a.url, rows[0]["prompt"], 32)  # warm request, not counted
    out = []
    for rd in range(a.rounds):
        for row in rows:
            r = req(a.url, row["prompt"], a.max_tokens)
            s = stats(a.url)
            last = s["stats"][-1] if s.get("stats") else {}
            if isinstance(last, dict):
                for k in ("accept_mean", "tokens_per_step", "steps"):
                    r[k] = last.get(k)
            r.update(round=rd, id=row["id"])
            out.append(r)
    os.makedirs(W, exist_ok=True)
    json.dump(out, open(f"{W}/speed-{a.tag}.json", "w"), indent=1)
    summ(a.tag, out)


def cmp(a):
    A = json.load(open(f"{W}/speed-{a.tagA}.json"))
    B = json.load(open(f"{W}/speed-{a.tagB}.json"))
    same = 0
    for x, y in zip(A, B):
        s = x["text"] == y["text"]
        same += s
        print(f"{x['id']} r{x['round']} same={s} accept {x.get('accept_mean')} / {y.get('accept_mean')} tok/s {x['tps']:.1f} / {y['tps']:.1f}")
    print(f"same text {same}/{len(A)}")
    summ(a.tagA, A)
    summ(a.tagB, B)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="c", required=True)
r = sp.add_parser("run")
r.add_argument("tag")
r.add_argument("--url", default="http://127.0.0.1:8087")
r.add_argument("--rounds", type=int, default=2)
r.add_argument("--max-tokens", type=int, default=256)
c = sp.add_parser("cmp")
c.add_argument("tagA")
c.add_argument("tagB")
a = ap.parse_args()
(run if a.c == "run" else cmp)(a)
