# SPDX-License-Identifier: Apache-2.0
"""Steady-state total decode throughput with N streams all decoding (and the burst aggregate incl. prefills).

One burst = N concurrent streaming requests (check/prompts.json, prompts 0..N-1, greedy, no thinking).
- steady window = [last stream's first token, first stream's last token]: every stream is past its prefill and none
  has finished. Tokens inside the window per stream = /tokenize of the text pieces that arrived inside it.
  steady_total_tps = sum of those tokens / window length.
- aggregate_tps = all completion tokens (usage) / burst wall time (incl. TTFT and the serialized prefills).
  python3 check/steady.py --url http://127.0.0.1:8087 --n 4 --rounds 3 --max-tokens 512 --out runs/steady-4.json
"""
import argparse, json, os, statistics, threading, time, urllib.request

def post(url, path, body, stream=False):
    return urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}), timeout=1800)

def one(url, prompt, mt, res, i):
    body = {"model": "qwen38", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); pieces = []; ntok = None
    with post(url, "/v1/chat/completions", body) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                ntok = int(u["completion_tokens"])
            if d.get("choices"):
                c = d["choices"][0].get("delta", {}).get("content") or ""
                if c:
                    pieces.append((time.time(), c))
    res[i] = dict(t0=t0, t1=time.time(), pieces=pieces, ntok=ntok)

def ntoks(url, text):
    if not text:
        return 0
    with post(url, "/tokenize", {"prompt": text, "add_special_tokens": False}) as r:
        return json.load(r)["count"]

def burst(url, rows, n, mt):
    res = [None] * n
    th = [threading.Thread(target=one, args=(url, rows[i % len(rows)]["prompt"], mt, res, i)) for i in range(n)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ws = max(r["pieces"][0][0] for r in res)
    we = min(r["pieces"][-1][0] for r in res)
    win = we - ws
    tok = sum(ntoks(url, "".join(c for t, c in r["pieces"] if ws < t <= we)) for r in res)
    total = sum(r["ntok"] for r in res)
    return dict(streams=n, steady_window_s=round(win, 3), steady_tokens=tok,
                steady_total_tps=round(tok / win, 1) if win > 0.5 else None,
                steady_per_stream_tps=round(tok / win / n, 2) if win > 0.5 else None,
                aggregate_tps=round(total / wall, 1), wall_s=round(wall, 2), completion_tokens=[r["ntok"] for r in res],
                ttft_s=[round(r["pieces"][0][0] - r["t0"], 3) for r in res])

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8087")
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"))
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--out", default="")
a = ap.parse_args()
rows = json.load(open(a.prompts))
burst(a.url, rows, a.n, a.max_tokens)  # warm-up, not counted
rs = [burst(a.url, rows, a.n, a.max_tokens) for _ in range(a.rounds)]
for r in rs:
    print(json.dumps(r))
st = [r["steady_total_tps"] for r in rs if r["steady_total_tps"]]
print(f"STEADY n={a.n}: steady_total_tps median {statistics.median(st) if st else None} {st}; aggregate_tps {[r['aggregate_tps'] for r in rs]}")
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rs, open(a.out, "w"), indent=1)
