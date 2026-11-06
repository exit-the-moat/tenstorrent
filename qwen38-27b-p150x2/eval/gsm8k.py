#!/usr/bin/env python3
"""GSM8K quality check: the first 60 questions of the openai/gsm8k "main" test split, greedy, thinking off.

    gsm8k.py fetch                       -> eval/gsm8k60.jsonl (once; already in the repo)
    gsm8k.py run --config <label> --precision <p> [--url ...] [--workers 4] [--max-tokens 2048]

Score = exact match of the last number in the answer with the reference after "####". Writes one row to
$EP1_EVAL_DIR/quality.csv (default ../work/eval) and every answer to <run dir>/answers.jsonl. Standard library only.
"""
import argparse, csv, datetime, json, os, re, sys, threading, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET = os.path.join(HERE, "gsm8k60.jsonl")  # the fixed subset, shipped with the repo
OUT = os.environ.get("EP1_EVAL_DIR") or os.path.join(os.environ.get("WORK") or os.path.join(HERE, "..", "work"), "eval")  # runs/, quality.csv
PROMPT = "Solve this math problem. Show your working briefly, then give the final answer on its own line as '#### <number>'.\n\n"


def fetch():
    os.makedirs(os.path.dirname(SUBSET), exist_ok=True)
    u = "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=test&offset=0&length=60"
    rows = json.load(urllib.request.urlopen(u))["rows"]
    with open(SUBSET, "w") as f:
        for r in rows:
            f.write(json.dumps({"idx": r["row_idx"], "question": r["row"]["question"], "answer": r["row"]["answer"]}) + "\n")
    print(f"DONE {len(rows)} questions")


def last_number(s):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"(-?[\d,]*\.?\d+)", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def norm(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run(a):
    qs = [json.loads(l) for l in open(SUBSET)]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(OUT, "runs", f"{stamp}-{a.config}")
    os.makedirs(run_dir, exist_ok=True)
    out = [None] * len(qs)
    lock = threading.Lock()
    nxt = iter(range(len(qs)))

    def worker():
        while True:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            q = qs[i]
            body = {"model": "qwen38", "messages": [{"role": "user", "content": PROMPT + q["question"]}],
                    "max_tokens": a.max_tokens, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            try:
                req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=3600))
                text = r["choices"][0]["message"].get("content") or ""
                fin = r["choices"][0].get("finish_reason")
            except Exception as e:
                text, fin = "", "error: " + repr(e)
            ref = q["answer"].split("####")[-1].strip().replace(",", "")
            got = last_number(text)
            out[i] = dict(idx=q["idx"], ref=ref, got=got, correct=norm(got) is not None and norm(got) == norm(ref),
                          finish=fin, text=text)
            print(f"[{sum(o is not None for o in out)}/{len(qs)}] q{q['idx']} ref={ref} got={got}", flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts: t.start()
    for t in ts: t.join()
    with open(os.path.join(run_dir, "answers.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    n_ok = sum(o["correct"] for o in out)
    n_cut = sum(o["finish"] == "length" for o in out)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "quality.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "config", "precision", "benchmark", "correct", "total", "score_pct",
                                          "cut_by_length", "run_dir"])
        if new:
            w.writeheader()
        w.writerow(dict(date=datetime.datetime.now().isoformat(timespec="seconds"), config=a.config,
                        precision=a.precision, benchmark="gsm8k-test-first60-greedy-nothink", correct=n_ok,
                        total=len(out), score_pct=round(100 * n_ok / len(out), 1), cut_by_length=n_cut, run_dir=run_dir))
    print(f"DONE {n_ok}/{len(out)} correct ({100 * n_ok / len(out):.1f} %), {n_cut} cut by length", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--url", default="http://127.0.0.1:8085")
        ap.add_argument("--config", required=True)
        ap.add_argument("--precision", required=True)
        ap.add_argument("--workers", type=int, default=4)
        ap.add_argument("--max-tokens", type=int, default=2048)
        run(ap.parse_args())
