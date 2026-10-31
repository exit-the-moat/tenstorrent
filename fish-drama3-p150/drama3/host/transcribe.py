# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Listen-free sanity checks of rendered clips: duration, level, clipping, and a faster-whisper (CPU) transcript.

Runs in its own process with /opt/ep0/pyext-whisper on the path (its huggingface_hub / tokenizers must not
shadow the ones transformers uses in the device process).

    PYTHONPATH=/opt/ep0/pyext-whisper:$PYTHONPATH python -m models.demos.blackhole.drama3.host.transcribe \
        /opt/ep0/day3/out/close.wav [...]  [--expect "text"]
"""

import argparse
import difflib
import json
import os
import re

import numpy as np
import soundfile as sf

MODEL = os.environ.get("DRAMA3_WHISPER", "/opt/ep0/models/faster-whisper-small")


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def check(path, model, expect=None):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    a = audio.mean(1)
    rms = float(np.sqrt(np.mean(a**2)) + 1e-12)
    res = dict(
        file=str(path),
        seconds=round(len(a) / sr, 2),
        sample_rate=sr,
        peak=round(float(np.abs(a).max()), 4),
        rms_dbfs=round(20 * np.log10(rms), 1),
        clipped_samples=int((np.abs(a) >= 0.999).sum()),
    )
    # faster-whisper wants 16 kHz mono float32
    import librosa

    a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
    segs, _ = model.transcribe(a16, language="en", beam_size=5)
    res["transcript"] = " ".join(s.text.strip() for s in segs).strip()
    if expect:
        ref, hyp = words(expect), words(res["transcript"])
        res["word_match_ratio"] = round(difflib.SequenceMatcher(None, ref, hyp).ratio(), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect", nargs="*", default=None, help="expected text per file")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    out = []
    for i, f in enumerate(a.files):
        out.append(check(f, model, a.expect[i] if a.expect else None))
        print(json.dumps(out[-1]), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
