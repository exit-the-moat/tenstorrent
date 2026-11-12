#!/usr/bin/env python3
"""Summarise one counted queue from <out>/runs.jsonl (rows with queue == <claim dir>): wall clock from the first job start
to the last image saved, images per hour, per-stage means. Appends one line to <out>/queue.jsonl and prints it.
Usage: qsum.py <out dir> <claim dir as gen.py saw it> <config label> <rep>   (standard library only; run.sh calls it)"""
import json, os, statistics as S, sys, time
out, q, label, rep = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [r for r in map(json.loads, open(os.path.join(out, "runs.jsonl"))) if r.get("queue") == q]
assert rows, "no rows for " + q
t0, t1 = min(r["t_start"] for r in rows), max(r["t_end"] for r in rows)
wall = t1 - t0
Q = dict(config=label, rep=rep, queue=q, images=len(rows), wall_s=round(wall, 1), images_per_hour=round(len(rows) * 3600 / wall, 2),
         s_per_image_mean_job=round(S.mean(r["seconds_total"] for r in rows), 2), encoder_mean=round(S.mean(r["seconds_encoder"] for r in rows), 2),
         denoising_mean=round(S.mean(r["seconds_denoising"] for r in rows), 2), vae_mean=round(S.mean(r["seconds_vae"] for r in rows), 2),
         step_mean=round(S.mean(r["seconds_per_step"] for r in rows), 3), processes=sorted({r.get("visible", "all") for r in rows}),
         start=time.strftime("%H:%M:%S", time.localtime(t0)), end=time.strftime("%H:%M:%S", time.localtime(t1)), tags=sorted({r["tag"] for r in rows}))
with open(os.path.join(out, "queue.jsonl"), "a") as f:
    f.write(json.dumps(Q) + "\n")
print(json.dumps(Q))
