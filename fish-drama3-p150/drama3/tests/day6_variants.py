# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 6 retry (team-lead list): score <dir>/<id>_<variant>_s<seed>.wav with two Whisper models vs the written form,
pick the best take per line, write <dir>/picks.json.

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day6_variants <dir> <variants.json>
"""
import difflib
import json
import re
import sys
from pathlib import Path

from models.demos.blackhole.drama3.host import transcribe as T

MODELS = {"small": "/opt/ep0/models/faster-whisper-small", "base": "/opt/ep0/models/faster-whisper-base"}
WATCH = ("moat", "mode", "mote", "bills", "builds", "felt", "felled", "card", "cart", "dead", "debt")


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def main():
    from faster_whisper import WhisperModel

    d = Path(sys.argv[1])
    variants = {v["id"]: v for v in json.load(open(sys.argv[2]))}
    ms = {k: WhisperModel(p, device="cpu", compute_type="int8", cpu_threads=8) for k, p in MODELS.items()}
    takes = []
    for wav in sorted(d.glob("*_s[0-9]*.wav")):
        vid, seed = wav.stem.rsplit("_s", 1)
        v = variants[vid]
        t = dict(file=wav.name, line=v["line"], variant=v["variant"], seed=int(seed), spoken=v["spoken"])
        for k, m in ms.items():
            r = T.check(wav, m, v["spoken"])
            t[f"{k}_transcript"] = r["transcript"]
            t[f"{k}_match"] = round(difflib.SequenceMatcher(None, words(v["written"]), words(r["transcript"])).ratio(), 3)
            t[f"{k}_heard"] = [w for w in words(r["transcript"]) if w in WATCH]
        t["clipped"] = r["clipped_samples"]
        takes.append(t)
        print(json.dumps(t), flush=True)
    picks = {}
    for line in sorted({t["line"] for t in takes}):
        ts = [t for t in takes if t["line"] == line]
        best = max(ts, key=lambda t: (t["small_match"] + t["base_match"], t["small_match"], -t["seed"]))
        picks[line] = dict(best=best["file"], variant=best["variant"], seed=best["seed"], small_match=best["small_match"],
                           base_match=best["base_match"], takes=ts)
        print(f"{line}: best {best['file']} small {best['small_match']} base {best['base_match']}", flush=True)
    json.dump(picks, open(d / "picks.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
