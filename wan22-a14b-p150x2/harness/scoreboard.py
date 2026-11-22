"""Episode 4: /out/results.csv from /out/runs.jsonl (generated, never typed).

One row per counted job. Runs whose name contains "contaminated" or "dev" are skipped (host was shared with other
work, or the run was functional only). Columns: config, seconds per clip, split, minutes of video per hour.
Video length = frames / 16 fps (Wan 2.2 outputs 16 fps; 81 frames = 5.06 s).
"""
import csv
import json

rows = [json.loads(line) for line in open("/out/runs.jsonl")]
out = []
for r in rows:
    if "contaminated" in r["run"] or r["run"].startswith("dev"):
        continue
    vid_s = r["frames"] / 16.0
    total = r["total_s"]
    env = r.get("env", {})
    out.append(dict(
        time=r["t"], run=r["run"], job=r["job"], seed=r["seed"], res=r["res"], frames=r["frames"], steps=r["steps"],
        cards=r["mesh"][0] * r["mesh"][1], split=r["split"], links=r["links"], dynamic_load=r["dynamic_load"],
        weights=env.get("EP4_DIT_WEIGHTS", "bf16"), quant=r.get("quant") or "", traced=r.get("traced", 0),
        levers=" ".join(f"{k}={v}" for k, v in sorted(env.items()) if k not in ("EP4_DIT_WEIGHTS", "TT_DIT_CACHE_DIR")),
        s_per_clip=round(total, 1), encoder_s=r["encoder_s"], denoise_s=r["denoise_s"], s_per_step=r["step_s"],
        vae_s=r["vae_s"], video_min_per_hour=round(60 * vid_s / total, 2), dram_mb=r["dram_mb"],
    ))
with open("/out/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
for o in out:
    print(o["run"], o["job"], o["res"], o["cards"], o["weights"], o["levers"], o["s_per_clip"], o["s_per_step"], o["video_min_per_hour"])
