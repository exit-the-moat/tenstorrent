"""Summarise the LTX-2.5 RTX PRO 6000 run: per clip parts, GPU energy (nvidia-smi, 0.5 s samples) and peak memory.
Writes results_gpu.csv next to this file. Usage: python analyse.py"""
import csv, glob, json, os
from datetime import datetime, timezone

H = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(H, "out", "out")
P = []
for r in csv.reader(open(os.path.join(O, "power.csv"))):
    if not r or r[0].startswith("timestamp"):
        continue
    try:
        t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        P.append((t, float(r[1].split()[0]), int(r[2].split()[0]), int(r[3]), int(r[5].split()[0])))
    except Exception:
        pass


def window(t0, t1):
    s = [p for p in P if t0 <= p[0] <= t1]
    if not s:
        return None
    wh = 0.0
    for a, b in zip(s, s[1:]):
        wh += (a[1] + b[1]) / 2 * (b[0] - a[0]) / 3600
    busy = [p for p in s if p[1] > 100]
    return dict(n=len(s), wh=round(wh, 2), w_mean=round(sum(p[1] for p in s) / len(s)),
                w_busy=round(sum(p[1] for p in busy) / len(busy)) if busy else None,
                w_max=round(max(p[1] for p in s)), sm_busy=round(sum(p[2] for p in busy) / len(busy)) if busy else None,
                tmax=max(p[3] for p in s), mem_max_gb=round(max(p[4] for p in s) / 1024, 1))


rows = []
for f in sorted(glob.glob(os.path.join(O, "*", "timing.jsonl"))):
    run = os.path.basename(os.path.dirname(f))
    for line in open(f):
        d = json.loads(line)
        if d.get("oom"):
            rows.append(dict(run=run, job="OOM", label=json.dumps(d)[:200])); continue
        w = window(d["t_wall_start"], d["t_wall_end"]) or {}
        p = d["parts"]
        rows.append(dict(run=run, job=d["job"], seed=d["seed"], label=d["label"], counted=d["counted"],
                         res=f'{d["width"]}x{d["height"]}', frames=d["frames"], retain=d["retain"], vae=d["vae"],
                         total_s=d["total_s"], render_s=d["render_s"], prompt=p.get("prompt_encode"),
                         img=p.get("image_encode"), s1=p.get("stage1"), ups=p.get("upsample"), s2=p.get("stage2"),
                         vdec=p.get("video_decode"), adec=p.get("audio_decode"), other=d["other_s"],
                         peak_alloc_gb=d["peak_alloc_gb"], peak_reserved_gb=d["peak_reserved_gb"],
                         smi_mem_max_gb=w.get("mem_max_gb"), gpu_wh=w.get("wh"), w_mean=w.get("w_mean"),
                         w_busy=w.get("w_busy"), w_max=w.get("w_max"), sm_mhz=w.get("sm_busy"), tmax=w.get("tmax"),
                         mp4=d["mp4"]))
for line in open(os.path.join(O, "cold.jsonl")):
    d = json.loads(line)
    w = window(d["t0"], d["t1"]) or {}
    rows.append(dict(run=d["label"], job="motion", seed=101, label="cold CLI", counted=True, frames=121,
                     total_s=d["wall_s"], gpu_wh=w.get("wh"), w_busy=w.get("w_busy"), smi_mem_max_gb=w.get("mem_max_gb"),
                     tmax=w.get("tmax")))
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(H, "results_gpu.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print({k: v for k, v in r.items() if v is not None and k not in ("mp4",)})
