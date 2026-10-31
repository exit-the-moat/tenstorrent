# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: listen-free checks (faster-whisper) of rendered dirs + per-line timing summary.

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_day4 \
        /opt/ep0/day4/out /opt/ep0/day4/out_v2
"""
import difflib
import json
import os
import re
import sys
from pathlib import Path

import yaml

from models.demos.blackhole.drama3.host import transcribe as T

# DRAMA3_LINES_FILE: a JSON list of {id, spoken, written} (Day 6 card_lines.json) instead of lines.yaml
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
        res, tot_a, tot_c, frames, bb, tw = [], 0.0, 0.0, 0, 0.0, 0.0
        for l in LINES:
            r = T.check(d / f"{l['id']}.wav", m, l["text"])
            r["id"] = l["id"]
            r["match_written"] = round(difflib.SequenceMatcher(None, words(l["written"]), words(r["transcript"])).ratio(), 3)
            t = timing[l["id"]]
            r.update({k: t.get(k) for k in ("frames", "compute_s", "rtf", "ttfa_s", "prefill_s", "codec_tail_s", "power_w", "board_power_w")})
            r["backbone_ms"] = round(t["backbone_ms_per_frame"], 1)
            r["tower_ms"] = round(t["tower_ms_per_frame"], 1)
            res.append(r)
            tot_a += t["audio_s"]
            tot_c += t["compute_s"]
            frames += t["frames"]
            bb += t["decode_backbone_s"]
            tw += t["tower_steps_s"]
            print(
                f"{d.name:7s} {l['id']:15s} {r['seconds']:5.2f}s fr {r['frames']:3d} rtf {r['rtf']:4.2f} ttfa {r['ttfa_s']:4.2f} "
                f"bb {r['backbone_ms']:4.1f} tw {r['tower_ms']:4.1f} pw {r['power_w']} clip {r['clipped_samples']} "
                f"match {r['match_written']:.3f} | {r['transcript']}",
                flush=True,
            )
        json.dump(res, open(d / "sanity.json", "w"), indent=1)
        print(
            f"{d.name}: audio {tot_a:.1f}s compute {tot_c:.1f}s overall RTF {tot_c / tot_a:.2f}; "
            f"backbone {1e3 * bb / frames:.1f} ms/frame, tower {1e3 * tw / frames:.1f} ms/frame",
            flush=True,
        )


if __name__ == "__main__":
    main()
