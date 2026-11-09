#!/usr/bin/env python3
"""Token-exact check: 5 fixed prompts, greedy, thinking off, exactly 128 tokens out each.

    exact.py capture NAME [--url http://127.0.0.1:8087]  -> <dir>/NAME.json (token ids per prompt)
    exact.py compare A B                                 -> per prompt: identical, or the first differing token

NAME = a file in the output directory (env CHECK_DIR, default ./runs); A/B may also be a path to a .json file, for
example ../reference/2card-exact.json. A change that keeps the maths passes with 5/5 identical. Standard library only.
"""
import json
import os
import sys
import urllib.request

PROMPTS = [
    "Explain in plain words what a hash map is and when it is slower than a list.",
    "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat.",
    "Review this code and list the bugs:\n\ndef mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n        total += xs[i]\n    return total / len(xs)\n",
    "Tell a short story about a lighthouse keeper who collects broken radios.",
    "What is 17 * 23 + 144 / 12? Show the steps.",
]
OUT = 128


def path(name):
    if name.endswith(".json") or "/" in name:
        return name
    d = os.environ.get("CHECK_DIR", "runs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".json")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def capture(name, base):
    res = []
    for p in PROMPTS:
        r = post(base + "/v1/chat/completions", {
            "model": "qwen38", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": OUT,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}, "stream": False})
        text = r["choices"][0]["message"].get("content") or ""
        reasoning = r["choices"][0]["message"].get("reasoning_content") or ""
        ids = post(base + "/tokenize", {"model": "qwen38", "prompt": reasoning + text, "add_special_tokens": False})["tokens"]
        res.append({"prompt": p, "text": reasoning + text, "ids": ids})
    json.dump(res, open(path(name), "w"), indent=1)
    print(f"captured {len(res)} prompts -> {path(name)}")


def compare(ref, new):
    a = json.load(open(path(ref)))
    b = json.load(open(path(new)))
    same = 0
    for k, (x, y) in enumerate(zip(a, b)):
        if x["ids"] == y["ids"]:
            same += 1
            print(f"prompt {k}: identical ({len(x['ids'])} tokens)")
        else:
            i = next((i for i, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), min(len(x["ids"]), len(y["ids"])))
            print(f"prompt {k}: DIFFERS at token {i} of {len(x['ids'])}")
    print(f"EXACT {same}/{len(a)}")
    return same == len(a)


if __name__ == "__main__":
    url = "http://127.0.0.1:8087"
    if "--url" in sys.argv:
        url = sys.argv[sys.argv.index("--url") + 1]
    if sys.argv[1] == "capture":
        capture(sys.argv[2], url)
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
