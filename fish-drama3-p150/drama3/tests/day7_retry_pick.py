# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 7: score the narration retries (<retry>/<id>_<variant>_s<seed>.wav) and the canonical seed-0 takes
(<narration>/<id>.wav) with two Whisper models (small + base, multilingual, language en: the only ones on the
server) and recommend one take per line. Writes [out json] (default <retry>/picks_day7.json); never touches the
narration dir.

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day7_retry_pick \
        /opt/ep0/day6/narration_retry/variants.json /opt/ep0/day6/narration /opt/ep0/day6/narration_retry [out json]

Ranking per line: the key words heard by both models (moat, cloud bills, where the line has them), then the mean
match vs the written form over both models, then the lower seed (seed 0 = the canonical take wins ties).
"""
import difflib
import json
import re
import sys
from pathlib import Path

from models.demos.blackhole.drama3.host import transcribe as T

MODELS = {"small": "/opt/ep0/models/faster-whisper-small", "base": "/opt/ep0/models/faster-whisper-base"}


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def match(written, hyp):
    return round(difflib.SequenceMatcher(None, words(written), words(hyp)).ratio(), 3)


def heard(phrase, *transcripts):
    return all(f" {phrase} " in f" {' '.join(words(t))} " for t in transcripts)


def main():
    from faster_whisper import WhisperModel

    lines_file, nar, retry = map(Path, sys.argv[1:4])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else retry / "picks_day7.json"
    variants = json.load(open(lines_file))
    written = {v["line"]: v["written"] for v in variants}
    takes = [dict(id=lid, variant="canonical", seed=0, file=str(nar / f"{lid}.wav")) for lid in written]
    for v in variants:
        for wav in sorted(retry.glob(f"{v['id']}_s*.wav")):
            takes.append(dict(id=v["line"], variant=v["variant"], seed=int(wav.stem.rsplit("_s", 1)[1]), file=str(wav)))
    for name, path in MODELS.items():
        m = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=8)
        for t in takes:
            r = T.check(t["file"], m, None)
            t[f"transcript_{name}"] = r["transcript"]
            t.setdefault("seconds", r["seconds"])
            t.setdefault("clipped_samples", r["clipped_samples"])
        del m
    for t in takes:
        w = written[t["id"]]
        ts, tb = t["transcript_small"], t["transcript_base"]
        t["match_vs_written"] = match(w, ts)  # small, as in the Day 6 sanity.json
        t["match_vs_written_base"] = match(w, tb)
        t["moat_heard"] = heard("moat", ts, tb) if "moat" in w.lower() else None
        t["moat_heard_small"], t["moat_heard_base"] = (heard("moat", ts), heard("moat", tb)) if "moat" in w.lower() else (None, None)
        t["cloud_bills_heard"] = heard("cloud bills", ts, tb) if "cloud bills" in w.lower() else None
        print(
            f"{t['id']} {t['variant']:9s} s{t['seed']} small {t['match_vs_written']:.3f} base {t['match_vs_written_base']:.3f} "
            f"moat {t['moat_heard']} bills {t['cloud_bills_heard']} | S: {ts} | B: {tb}",
            flush=True,
        )

    def key(t):
        kw = sum(int(bool(x)) for x in (t["moat_heard_small"], t["moat_heard_base"], t["cloud_bills_heard"]))
        return (kw, (t["match_vs_written"] + t["match_vs_written_base"]) / 2, -t["seed"], t["variant"] == "canonical")

    rec = {}
    for lid in written:
        best = max((t for t in takes if t["id"] == lid), key=key)
        rec[lid] = dict(variant=best["variant"], seed=best["seed"], file=best["file"], match_small=best["match_vs_written"],
                        match_base=best["match_vs_written_base"], moat_heard=best["moat_heard"],
                        cloud_bills_heard=best["cloud_bills_heard"], transcript_small=best["transcript_small"],
                        transcript_base=best["transcript_base"])
        print(f"RECOMMEND {lid}: {best['variant']} s{best['seed']} ({best['file']})", flush=True)
    json.dump(dict(models={k: v for k, v in MODELS.items()}, note="multilingual small + base with language=en "
                   "(no .en models on the server); match = word SequenceMatcher ratio vs the written form; *_heard = "
                   "heard by both models", takes=takes, recommended=rec), open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
