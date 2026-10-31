"""port6: chip / board power over a run window. power_sum.py <power.jsonl> <timing.json> -> adds _power to timing.json
(window: _summary start/end epochs for cbatch runs, else the earliest start .. latest end of the jobs)."""
import json, sys
s = [json.loads(l) for l in open(sys.argv[1]) if "\"power\"" in l]
tp = sys.argv[2]
t = json.load(open(tp))
if "_summary" in t:
    t0, t1 = t["_summary"]["start_epoch"], t["_summary"]["end_epoch"]
else:
    js = [v for k, v in t.items() if isinstance(v, dict) and "start_epoch" in v]
    t0, t1 = min(v["start_epoch"] for v in js), max(v["end_epoch"] for v in js)
w = [x for x in s if t0 <= x["t"] <= t1]
p = dict(samples=len(w))
if w:
    for k in ("power", "board_power"):
        v = [x[k] for x in w]
        p[k] = dict(mean=round(sum(v) / len(v), 1), min=round(min(v), 1), max=round(max(v), 1))
t["_power"] = p
json.dump(t, open(tp, "w"), indent=1)
print(json.dumps(p))
