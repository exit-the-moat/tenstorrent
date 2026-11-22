"""Episode 4: energy per clip from the passive power log (hwmon, 5 s) and runs.jsonl.

Clip window = [end - total_s, end] (end = runs.jsonl time, UTC). Per clip: mean card W (both cards, hwmon
`blackhole` power1), mean PSU W (Corsair HX1500i output power: the WHOLE box incl. host, disks and the other
services on this server), Wh per clip, kWh per minute of video, EUR at --eur-kwh (default 0.35 = Sebastian's tariff, 17 Nov
gives his tariff). Writes /out/power_summary.csv.
  python3 powersum.py logs/power_chain4.csv [--eur-kwh 0.30]
"""
import argparse
import calendar
import csv
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--eur-kwh", type=float, default=0.35)  # Sebastian's tariff, 17 Nov
a = ap.parse_args()

samples = []
for line in open(a.log):
    if line.startswith("#") or line.startswith("epoch"):
        continue
    f = line.strip().split(",")
    if len(f) < 12:
        continue
    samples.append((float(f[0]), float(f[1]), float(f[6]), float(f[11]), float(f[5]), float(f[10])))
if not samples:
    raise SystemExit("no samples")
t0, t1 = samples[0][0], samples[-1][0]
rows = []
for line in open("/out/runs.jsonl"):
    r = json.loads(line)
    end = calendar.timegm(time.strptime(r["t"], "%Y-%m-%d %H:%M:%S"))
    start = end - r["total_s"]
    if start < t0 or end > t1:
        continue
    win = [s for s in samples if start <= s[0] <= end]
    if len(win) < 3:
        continue
    c0 = sum(s[1] for s in win) / len(win)
    c1 = sum(s[2] for s in win) / len(win)
    psu = sum(s[3] for s in win) / len(win)
    clk = sum((s[4] + s[5]) / 2 for s in win) / len(win)
    vid_min = r["frames"] / 16.0 / 60.0
    wh_cards = (c0 + c1) * r["total_s"] / 3600
    wh_psu = psu * r["total_s"] / 3600
    rows.append(dict(run=r["run"], job=r["job"], res=r["res"], total_s=r["total_s"], samples=len(win),
                     card0_W=round(c0, 1), card1_W=round(c1, 1), cards_W=round(c0 + c1, 1), psu_W=round(psu, 1),
                     aiclk_MHz=round(clk), Wh_cards=round(wh_cards, 2), Wh_psu=round(wh_psu, 2),
                     kWh_per_video_min_psu=round(wh_psu / 1000 / vid_min, 3),
                     eur_per_clip_psu=round(wh_psu / 1000 * a.eur_kwh, 4),
                     eur_per_video_s_psu=round(wh_psu / 1000 * a.eur_kwh / (r["frames"] / 16.0), 5),
                     eur_kwh_sebastian_17nov=a.eur_kwh))
if rows:
    tag = a.log.rsplit("/", 1)[-1].replace("power_", "").replace(".csv", "")
    outname = "/out/power_summary.csv" if tag == "chain4" else f"/out/power_summary_{tag}.csv"
    with open(outname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
for r in rows:
    print(r["run"], r["job"], r["res"], r["total_s"], "cards", r["cards_W"], "W psu", r["psu_W"], "W", r["Wh_psu"], "Wh/clip",
          r["eur_per_clip_psu"], "EUR/clip @", r["eur_kwh_sebastian_17nov"])
