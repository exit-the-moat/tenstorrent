"""Summary table of the rented-GPU run from the fetched files (timing.json per pass, smi.csv, load.json).

    python summarize.py <results dir>   -> prints a markdown table and writes summary.json
Power and memory = nvidia-smi samples (0.5 s) whose epoch falls inside a line's [start, end] window.
"""
import csv
import json
import statistics as stx
import sys
from pathlib import Path

d = Path(sys.argv[1])
smi = [dict(zip(r.keys(), r.values())) for r in csv.DictReader(open(d / "smi.csv"))]
smi = [{k: float(v) for k, v in r.items()} for r in smi if r.get("power.draw") not in (None, "", "[N/A]")]


def window(t0, t1, key):
    return [r[key] for r in smi if t0 <= r["epoch"] <= t1]


def stats(vals):
    return {"mean": round(stx.mean(vals), 1), "min": round(min(vals), 1), "max": round(max(vals), 1), "n": len(vals)} if vals else None


out = {"load": json.load(open(d / "load.json")), "warmup": json.load(open(d / "warmup.json"))}
idle = [r for r in smi if r["epoch"] < out["warmup"]["start_epoch"]][-20:]
out["idle_after_load"] = {"power_w": stats([r["power.draw"] for r in idle]), "memory_mib": stats([r["memory.used"] for r in idle])}
for name in ("seq", "seq2"):
    t = json.load(open(d / name / "timing.json"))
    pw, mem = [], []
    for v in t.values():
        pw += window(v["start_epoch"], v["end_epoch"], "power.draw")
        mem += window(v["start_epoch"], v["end_epoch"], "memory.used")
    a = sum(v["audio_s"] for v in t.values())
    g = sum(v["gen_s"] for v in t.values())
    fr = sum(v["frames_usage"] or 0 for v in t.values())
    ids = list(t)
    out[name] = {
        "lines": len(t), "audio_s": round(a, 1), "gen_s": round(g, 2), "overall_rtf": round(g / a, 4),
        "rtf_per_line": [min(v["rtf"] for v in t.values()), max(v["rtf"] for v in t.values())],
        "ttfa_first_line_s": t[ids[0]]["ttfa_s"],
        "ttfa_later_lines_s": [min(t[i]["ttfa_s"] for i in ids[1:]), max(t[i]["ttfa_s"] for i in ids[1:])],
        "ttfa_all_median_s": round(stx.median(v["ttfa_s"] for v in t.values()), 4),
        "frames": fr, "ms_per_frame_whole_request": round(1e3 * g / fr, 2) if fr else None,
        "ms_per_frame_after_first_audio_median": round(stx.median(v["ms_per_frame_after_first_audio"] for v in t.values()), 2),
        "power_w_while_speaking": stats(pw), "memory_mib_while_speaking": stats(mem),
        "finish_reasons": sorted({str(v["finish_reason"]) for v in t.values()}),
    }
t = json.load(open(d / "c8" / "timing.json"))
s = t.pop("_summary")
s["rtf_per_stream"] = [min(v["rtf"] for v in t.values()), max(v["rtf"] for v in t.values())]
s["ttfa_per_stream_s"] = [min(v["ttfa_s"] for v in t.values()), max(v["ttfa_s"] for v in t.values())]
s["power_w"] = stats(window(s["start_epoch"], s["end_epoch"], "power.draw"))
s["memory_mib"] = stats(window(s["start_epoch"], s["end_epoch"], "memory.used"))
out["c8"] = s
json.dump(out, open(d / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
