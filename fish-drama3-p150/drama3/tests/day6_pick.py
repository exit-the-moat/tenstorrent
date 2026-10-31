# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 6: for the narration lines whose Whisper match vs the written form is < 0.85, score the re-renders
(<retry>/<id>_s<seed>.wav), keep the best take (ties keep seed 0) in the narration dir; the seed-0 take moves to
<retry>/<id>_s0.*. Writes <narration>/picks.json and updates timing.json / sanity.json there.

    DRAMA3_LINES_FILE=/opt/ep0/voice/card_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH \
        python -m models.demos.blackhole.drama3.tests.day6_pick /opt/ep0/day6/narration /opt/ep0/day6/retry
"""
import difflib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from models.demos.blackhole.drama3.host import transcribe as T


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def main():
    from faster_whisper import WhisperModel

    nar, retry = Path(sys.argv[1]), Path(sys.argv[2])
    lines = {l["id"]: l for l in json.load(open(os.environ.get("DRAMA3_LINES_FILE", "/opt/ep0/voice/card_lines.json")))}
    sanity = {r["id"]: r for r in json.load(open(nar / "sanity.json"))}
    timing = json.load(open(nar / "timing.json"))
    rt = json.load(open(retry / "timing.json"))
    m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    picks = {}
    for lid, r0 in sanity.items():
        if r0["match_written"] >= 0.85:
            continue
        takes = [(r0["match_written"], 0, r0)]
        for wav in sorted(retry.glob(f"{lid}_s*.wav")):
            seed = int(wav.stem.rsplit("_s", 1)[1])
            if seed == 0:
                continue
            r = T.check(wav, m, lines[lid]["spoken"])
            r["id"] = lid
            r["match_written"] = round(difflib.SequenceMatcher(None, words(lines[lid]["written"]), words(r["transcript"])).ratio(), 3)
            takes.append((r["match_written"], seed, r))
            print(f"{lid} s{seed}: {r['match_written']:.3f} | {r['transcript']}", flush=True)
        best = max(takes, key=lambda x: (x[0], -x[1]))
        picks[lid] = dict(seeds=[s for _, s, _ in takes], matches=[mm for mm, _, _ in takes], kept_seed=best[1], kept_match=best[0])
        print(f"{lid}: seed 0 {r0['match_written']:.3f} -> keep seed {best[1]} ({best[0]:.3f})", flush=True)
        if best[1] != 0:
            for ext in (".wav", ".frames.pt"):
                shutil.move(nar / f"{lid}{ext}", retry / f"{lid}_s0{ext}")
                shutil.copy(retry / f"{lid}_s{best[1]}{ext}", nar / f"{lid}{ext}")
            t = dict(rt[f"{lid}_s{best[1]}"], seed=best[1])
            for k in ("power_w", "board_power_w"):
                t.setdefault(k, timing[lid].get(k))
            timing[lid] = t
            r = best[2]
            r.update({k: t.get(k) for k in ("frames", "compute_s", "rtf", "ttfa_s", "prefill_s", "codec_tail_s", "power_w", "board_power_w")})
            r["backbone_ms"] = round(t["backbone_ms_per_frame"], 1)
            r["tower_ms"] = round(t["tower_ms_per_frame"], 1)
            r["seed"] = best[1]
            sanity[lid] = r
    json.dump(timing, open(nar / "timing.json", "w"), indent=1)
    json.dump([sanity[l] for l in lines if l in sanity], open(nar / "sanity.json", "w"), indent=1)
    json.dump(picks, open(nar / "picks.json", "w"), indent=1)
    print("still < 0.85:", {l: p["kept_match"] for l, p in picks.items() if p["kept_match"] < 0.85}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
