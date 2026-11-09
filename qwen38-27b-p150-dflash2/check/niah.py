#!/usr/bin/env python3
"""Needle-in-a-haystack check: a long context really is read to its end.

    niah.py NAME [--url http://127.0.0.1:8087] [--lengths 8192,32768] [--depths 10,50,90] [--filler DIR]

One needle sentence with a random 6-digit number at each depth (% of the context) in filler made of source code
(by default the qwen36/ folder of this package), sized with the server's own /tokenize. Greedy, thinking off, up to
24 tokens; a case passes if the answer contains the number. Prints NIAH <passed>/<total> and writes
<dir>/niah_NAME.json (env CHECK_DIR, default ./runs). Standard library only.
"""
import json
import os
import pathlib
import random
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    name = sys.argv[1]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    base = arg("--url", "http://127.0.0.1:8087")
    lengths = [int(x) for x in arg("--lengths", "8192,32768").split(",")]
    depths = [int(x) for x in arg("--depths", "10,50,90").split(",")]
    filler = pathlib.Path(arg("--filler", str(HERE.parent / "qwen36")))
    text = "\n".join(p.read_text(errors="ignore") for p in sorted(filler.rglob("*.py")))
    toks = post(base + "/tokenize", {"model": "qwen38", "prompt": text[:3_000_000], "add_special_tokens": False})["tokens"]
    if len(toks) < max(lengths):
        toks = toks * (max(lengths) // len(toks) + 1)
    rng = random.Random(7)
    res, ok = [], 0
    for n in lengths:
        body_n = n - 200  # room for the needle, the question and the chat template
        for d in depths:
            code = rng.randint(100000, 999999)
            needle = f"\n# NOTE: the secret access code for the vault is {code}. Remember it.\n"
            k = body_n * d // 100
            before = post(base + "/detokenize", {"model": "qwen38", "tokens": toks[:k]})["prompt"]
            after = post(base + "/detokenize", {"model": "qwen38", "tokens": toks[k:body_n]})["prompt"]
            prompt = (before + needle + after +
                      "\n\nQuestion: what is the secret access code for the vault mentioned in the text above? "
                      "Answer with the number only.")
            r = post(base + "/v1/chat/completions", {
                "model": "qwen38", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}, "stream": False})
            ans = (r["choices"][0]["message"].get("content") or "").strip()
            hit = str(code) in ans
            ok += hit
            res.append({"length": n, "depth": d, "code": code, "answer": ans, "pass": hit,
                        "prompt_tokens": r.get("usage", {}).get("prompt_tokens")})
            print(f"{n:>6} tokens, depth {d:>2} %: {'PASS' if hit else 'FAIL'} ({ans[:40]!r})", flush=True)
    d = os.environ.get("CHECK_DIR", "runs")
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, f"niah_{name}.json"), "w"), indent=1)
    print(f"NIAH {ok}/{len(res)}")


if __name__ == "__main__":
    main()
