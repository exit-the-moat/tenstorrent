# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4 quality gate: transcript, frame count (+-10 %) and code identity of re-rendered lines vs the Day 3 baseline.

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate <run dir> [ids]
"""
import json
import os
import sys
from pathlib import Path

import torch
import yaml

from models.demos.blackhole.drama3.host import transcribe as T

BASE = Path(os.environ.get("GATE_BASE", "/opt/ep0/day3/out"))  # GATE_BASE=/opt/ep0/day4/out_v2 for ref_v2 runs


def main():
    run = Path(sys.argv[1])
    ids = (sys.argv[2] if len(sys.argv) > 2 else "close,fix").split(",")
    lines = {l["id"]: l for l in yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]}
    from faster_whisper import WhisperModel

    m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    out = {}
    for i in ids:
        r = T.check(run / f"{i}.wav", m, lines[i]["text"])
        f, b = torch.load(run / f"{i}.frames.pt"), torch.load(BASE / f"{i}.frames.pt")
        n = min(len(f), len(b))
        r.update(
            frames=len(f),
            baseline_frames=len(b),
            frames_within_10pct=abs(len(f) - len(b)) <= 0.1 * len(b),
            identical_frames_prefix=int(next((k for k in range(n) if not torch.equal(f[k], b[k])), n)),
        )
        out[i] = r
        print(json.dumps(r), flush=True)
    json.dump(out, open(run / "gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
