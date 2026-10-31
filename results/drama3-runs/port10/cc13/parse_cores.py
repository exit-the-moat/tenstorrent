import csv, sys, json
from collections import defaultdict
log, names = sys.argv[1], sys.argv[2].split(",")
ev = []
with open(log) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    for r in csv.DictReader(f, skipinitialspace=True):
        if r["type"] == "ZONE_START" and r["zone name"].endswith("-KERNEL"):
            ev.append((int(r["time[cycles since reset]"]), int(r["run host ID"]), (int(r["core_x"]), int(r["core_y"]))))
ev.sort()
groups, cur = [], []
for e in ev:
    if cur and (e[0] - cur[-1][0]) / freq > 5000:
        groups.append(cur); cur = []
    cur.append(e)
groups.append(cur)
print("groups", len(groups), [len(g) for g in groups][-12:])
groups = groups[-len(names):]
out = {}
for n, g in zip(names, groups):
    prog = defaultdict(set)
    for t, rid, c in g:
        prog[rid].add(c)
    allc = set().union(*prog.values())
    xs = sorted({c[0] for c in allc}); ys = sorted({c[1] for c in allc})
    colp = {x: sum(1 for p in prog.values() if any(c[0] == x for c in p)) for x in xs}
    rowp = {y: sum(1 for p in prog.values() if any(c[1] == y for c in p)) for y in ys}
    print(n, "programs", len(prog), "cores", len(allc), "span_us", round((g[-1][0]-g[0][0])/freq,1))
    print("   progs per virtual x", colp)
    print("   progs per virtual y", rowp)
    out[n] = dict(prog={str(k): sorted(v) for k, v in prog.items()})
json.dump(out, open(sys.argv[3], "w"))
