# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 5b: Whisper check (vs the written form) + the throughput summary of a batch-mode run dir (host/batch.py).
Writes sanity.json and batch_summary.json into the dir.

    [DRAMA3_LINES_FILE=card_lines.json] PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH \
        python -m models.demos.blackhole.drama3.tests.summarize_batch <dir> [<dir> ...]
"""
import difflib
import json
import os
import re
import sys
from pathlib import Path

import yaml

from models.demos.blackhole.drama3.host import transcribe as T

if os.environ.get("DRAMA3_LINES_FILE"):
    LINES = [dict(l, text=l["spoken"]) for l in json.load(open(os.environ["DRAMA3_LINES_FILE"]))]
else:
    LINES = yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def main():
    from faster_whisper import WhisperModel

    m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    for d in map(Path, sys.argv[1:]):
        timing = json.load(open(d / "timing.json"))
        res = []
        for l in LINES:
            if l["id"] not in timing:
                continue
            r = T.check(d / f"{l['id']}.wav", m, l["text"])
            r["id"] = l["id"]
            r["match_written"] = round(difflib.SequenceMatcher(None, words(l["written"]), words(r["transcript"])).ratio(), 3)
            t = timing[l["id"]]
            r.update({k: t.get(k) for k in ("frames", "audio_s", "compute_s", "rtf", "ttfa_s", "prefill_s", "batch_index", "power_w", "board_power_w")})
            res.append(r)
            print(
                f"{d.name} {l['id']:15s} {r['seconds']:5.2f}s fr {r['frames']:3d} rtf {r['rtf']:4.2f} ttfa {r['ttfa_s']:5.2f} "
                f"b{r['batch_index']} pw {r['power_w']} clip {r['clipped_samples']} match {r['match_written']:.3f} | {r['transcript']}",
                flush=True,
            )
        json.dump(res, open(d / "sanity.json", "w"), indent=1)
        batches = [v for k, v in timing.items() if k.startswith("_batch") and k[6:].isdigit()]
        wall = sum(b["wall_s"] for b in batches)
        audio = sum(b["audio_s"] for b in batches)
        steps = [timing[l["id"]] for l in LINES if l["id"] in timing]
        pw = [json.loads(x) for x in open(d / "power.jsonl") if '"power"' in x] if (d / "power.jsonl").exists() else []
        t0 = min(t["start_epoch"] for t in steps)
        t1 = max(t["end_epoch"] for t in steps)
        pw = [x for x in pw if t0 <= x["t"] <= t1]
        summ = dict(
            batch_size=timing["_batch"]["batch_size"],
            card_dram_gb=timing["_batch"]["dram_gb"],
            lines=len(steps),
            batches=len(batches),
            audio_s=round(audio, 2),
            wall_s=round(wall, 2),
            audio_per_wall=round(audio / wall, 3),
            rtf_per_stream=[round(min(t["rtf"] for t in steps), 3), round(max(t["rtf"] for t in steps), 3)],
            ttfa_s=[round(min(t["ttfa_s"] for t in steps), 2), round(max(t["ttfa_s"] for t in steps), 2)],
            backbone_ms_per_step=round(sum(t["backbone_ms_per_step"] for t in steps) / len(steps), 1),
            tower_ms_per_step=round(sum(t["tower_ms_per_step"] for t in steps) / len(steps), 1),
            host_ms_per_step=round(sum(t["host_ms_per_step"] for t in steps) / len(steps), 1),
            chip_power_w=[round(min(x["power"] for x in pw), 1), round(max(x["power"] for x in pw), 1)] if pw else None,
            board_power_w=[round(min(x["board_power"] for x in pw), 1), round(max(x["board_power"] for x in pw), 1)] if pw else None,
            whisper_match_min=min(r["match_written"] for r in res),
            clipped=sum(r["clipped_samples"] for r in res),
        )
        json.dump(summ, open(d / "batch_summary.json", "w"), indent=1)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
