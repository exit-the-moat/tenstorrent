"""Whisper word match for the rented-GPU WAVs, the same way summarize_day4 scores the card's lines.

Run on the AI server inside the tt-metal env (it imports host.transcribe for the identical faster-whisper call):
    /opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/refgpu/score_refgpu.py /opt/ep0/refgpu/seq ..."
Writes <dir>/sanity.json: per line T.check(...) + match_written (difflib ratio vs the written form), which is the
"Whisper mean (min)" definition of day7_scoreboard.py.
"""
import difflib
import json
import re
import sys
from pathlib import Path

import yaml

from models.demos.blackhole.drama3.host import transcribe as T

LINES = yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def main():
    from faster_whisper import WhisperModel

    m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    for d in map(Path, sys.argv[1:]):
        res = []
        for l in LINES:
            r = T.check(d / f"{l['id']}.wav", m, l["text"])
            r["id"] = l["id"]
            r["match_written"] = round(difflib.SequenceMatcher(None, words(l["written"]), words(r["transcript"])).ratio(), 3)
            res.append(r)
            print(f"{d.name:5s} {l['id']:15s} {r['seconds']:5.2f}s match {r['match_written']:.3f} | {r['transcript']}", flush=True)
        json.dump(res, open(d / "sanity.json", "w"), indent=1)
        mw = [r["match_written"] for r in res]
        print(f"{d.name}: whisper mean {sum(mw) / len(mw):.3f} min {min(mw):.3f}", flush=True)


if __name__ == "__main__":
    main()
