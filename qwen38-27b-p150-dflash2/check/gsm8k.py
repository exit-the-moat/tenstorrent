#!/usr/bin/env python3
"""GSM8K quality check: the first 60 questions of the openai/gsm8k "main" test split (gsm8k60.jsonl), greedy, thinking off.

    gsm8k.py run --config LABEL [--url http://127.0.0.1:8087] [--workers 1] [--max-tokens 2048]

Score = exact match of the last number in the answer with the reference after "####". Prints the score and writes
<dir>/LABEL/answers.jsonl (env CHECK_DIR, default ./runs). Standard library only.
"""
import argparse
import json
import os
import pathlib
import re
import threading
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PROMPT = "Solve this math problem. Show your working briefly, then give the final answer on its own line as '#### <number>'.\n\n"


def last_number(s):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"(-?[\d,]*\.?\d+)", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def norm(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ask(url, q, mt):
    body = {"model": "qwen38", "messages": [{"role": "user", "content": PROMPT + q}], "temperature": 0, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    return j["choices"][0]["message"].get("content") or "", j["choices"][0].get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8087")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    qs = [json.loads(line) for line in open(HERE / "gsm8k60.jsonl")]
    out = [None] * len(qs)
    idx = iter(range(len(qs)))
    lock = threading.Lock()

    def work():
        while True:
            with lock:
                i = next(idx, None)
            if i is None:
                return
            text, fin = ask(a.url, qs[i]["question"], a.max_tokens)
            gold = qs[i]["answer"].split("####")[-1].strip().replace(",", "")
            pred = last_number(text)
            out[i] = {"idx": qs[i]["idx"], "pred": pred, "gold": gold, "ok": norm(pred) is not None and norm(pred) == norm(gold),
                      "finish": fin, "answer": text}

    ts = [threading.Thread(target=work) for _ in range(a.workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    d = pathlib.Path(os.environ.get("CHECK_DIR", "runs")) / a.config
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "answers.jsonl", "w") as f:
        for r_ in out:
            f.write(json.dumps(r_) + "\n")
    ok = sum(r_["ok"] for r_ in out)
    print(f"GSM8K {a.config}: {ok}/{len(out)} correct, {sum(r_['finish'] == 'length' for r_ in out)} cut by length")


if __name__ == "__main__":
    main()
