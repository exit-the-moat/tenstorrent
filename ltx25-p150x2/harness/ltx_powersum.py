"""ltx1: energy per clip from powerlog2 CSV + /out/runs.jsonl. usage: ltx_powersum.py <power.csv> <run name> [eur/kWh]"""
import csv, json, sys, time, calendar
pf, run = sys.argv[1], sys.argv[2]; eur = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
rows = [r for r in csv.DictReader(l for l in open(pf) if not l.startswith("#"))]
for r in rows:
    for k in r: r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
out = []
for l in open("/out/runs.jsonl"):
    j = json.loads(l)
    if j["run"] != run or not j["counted"]: continue
    t0 = calendar.timegm(time.strptime(j["start"], "%Y-%m-%d %H:%M:%S")); t1 = t0 + j["wall_s"]  # container clock is UTC; t1 = t0 + j["wall_s"]
    w = [r for r in rows if t0 <= r["epoch"] <= t1]
    if not w: continue
    m = lambda k: sum(r[k] for r in w) / len(w)
    cards = m("c0_W") + m("c1_W"); psu = m("psu_W")
    rec = dict(run=run, name=j["name"], tag=j["tag"], wall_s=j["wall_s"], samples=len(w), cards_W=round(cards, 1), c0_W=round(m("c0_W"), 1),
               c1_W=round(m("c1_W"), 1), aiclk0=round(m("c0_aiclk")), aiclk1=round(m("c1_aiclk")), psu_W=round(psu, 1),
               Wh_pc=round(psu * j["wall_s"] / 3600, 2), Wh_cards=round(cards * j["wall_s"] / 3600, 2),
               ct_pc=round(psu * j["wall_s"] / 3600 / 1000 * eur * 100, 3))
    out.append(rec); print(json.dumps(rec))
with open("/out/power_summary.jsonl", "a") as f:
    for r in out: f.write(json.dumps(r) + "\n")
