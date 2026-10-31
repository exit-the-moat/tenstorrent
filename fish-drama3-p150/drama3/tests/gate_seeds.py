# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: Whisper check of every <id>_s<seed>.wav in one or more run dirs (multi-seed runs): match ratio, frames,
rms, clipping per file, and per dir the mean match and frame counts.

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate_seeds <dir> [<dir> ...]
"""
import json
import sys
from pathlib import Path

import torch
import yaml

from models.demos.blackhole.drama3.host import transcribe as T


def main():
    lines = {l["id"]: l for l in yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]}
    from faster_whisper import WhisperModel

    m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    for d in map(Path, sys.argv[1:]):
        rows = []
        for wav in sorted(d.glob("*_s*.wav")):
            lid = wav.stem.rsplit("_s", 1)[0]
            r = T.check(wav, m, lines[lid]["text"])
            r["frames"] = len(torch.load(wav.with_suffix(".frames.pt")))
            r["id"] = wav.stem
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("id", "frames", "word_match_ratio", "rms_dbfs", "clipped_samples", "transcript")}), flush=True)
        for lid in sorted({r["id"].rsplit("_s", 1)[0] for r in rows}):
            rr = [r for r in rows if r["id"].startswith(lid + "_s")]
            print(json.dumps(dict(dir=str(d), line=lid, n=len(rr), mean_match=round(sum(r["word_match_ratio"] for r in rr) / len(rr), 3),
                                  frames=[r["frames"] for r in rr], clipped=sum(r["clipped_samples"] for r in rr))), flush=True)


if __name__ == "__main__":
    main()
