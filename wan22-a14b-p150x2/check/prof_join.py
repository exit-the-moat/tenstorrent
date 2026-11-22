"""Episode 4: per-op-family device time for the last DiT forward on one chip, joining tracy_ops_data.csv (op names by
global call count) with cpp_device_perf_report.csv (device kernel durations). Used when tracy's own report step fails.
  python3 prof_join.py <prof dir> [device] [n_forwards] [wall_s]
"""
import collections
import csv
import json
import re
import sys

d = sys.argv[1] + "/.logs/"
dev = sys.argv[2] if len(sys.argv) > 2 else "0"
nfw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
wall = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
names = {}
txt = open(d + "tracy_ops_data.csv").read()
for m in re.finditer(r'TT_DNN_DEVICE_OP: "([^"]+)", \d+, (\d+), \w+, (\d+)', txt):
    names[(m.group(2), int(m.group(3)))] = m.group(1)
rows = [r for r in csv.DictReader(open(d + "cpp_device_perf_report.csv")) if r["DEVICE ID"] == dev]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
rows = [r for r in rows if r["DEVICE KERNEL START CYCLE"]]
rows.sort(key=lambda r: int(r["DEVICE KERNEL START CYCLE"]))
tail = rows[-3000:]
gaps = [(int(tail[i + 1]["DEVICE KERNEL START CYCLE"]) - int(tail[i]["DEVICE KERNEL END CYCLE"]), i)
        for i in range(len(tail) - 1)]
g, gi = max(gaps)
last = tail[gi + 1:]
per = len(last)
print(f"boundary: largest gap {g/1.35e9:.2f} s before the last {per} ops")
fam = collections.defaultdict(lambda: [0.0, 0])
tot = 0.0
for r in last:
    ns = float(r["DEVICE KERNEL DURATION [ns]"] or 0)
    n = names.get((dev, int(r["GLOBAL CALL COUNT"])), "?")
    fam[n][0] += ns / 1e9
    fam[n][1] += 1
    tot += ns / 1e9
start = min(int(r["DEVICE KERNEL START CYCLE"]) for r in last)
end = max(int(r["DEVICE KERNEL END CYCLE"]) for r in last)
span = (end - start) / 1.35e9  # cycles at ~1.35 GHz AICLK (approximate; clock may throttle)
print(f"device {dev}: {len(rows)} ops total, last forward {per} ops, kernel sum {tot:.3f} s, span ~{span:.3f} s (cycles/1.35 GHz), wall {wall}")
out = []
for n, (s, c) in sorted(fam.items(), key=lambda x: -x[1][0]):
    print(f"  {s:7.3f} s  {100*s/tot:5.1f} %  {c:5d}  {n}")
    out.append(dict(op=n, s=round(s, 4), calls=c, share=round(s / tot, 4)))
json.dump(dict(device=dev, ops=per, kernel_sum_s=round(tot, 3), span_s=round(span, 3), families=out),
          open(sys.argv[1] + f"/summary_dev{dev}.json", "w"), indent=1)
