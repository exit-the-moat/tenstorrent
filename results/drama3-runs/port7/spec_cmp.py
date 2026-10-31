"""port7: spectral comparison of run dirs against a base dir (same 8 lines, different takes): per band mean log
power (dB, speech frames only), spectral flatness, clipping, rms; flags a band that moves by more than 3 dB on
average. python spec_cmp.py <base dir> <run dir> [...] -> prints one JSON line per run dir."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

BANDS = [(0, 500), (500, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 22050)]


def load(p):
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
        return x, w.getframerate()


def stats(p):
    x, sr = load(p)
    n, hop = 2048, 512
    win = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    e = spec.sum(-1)
    speech = e > np.percentile(e, 30)  # drop the quietest 30 % (pauses)
    s = spec[speech]
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi}"] = float(10 * np.log10(s[:, m].sum(-1) / s.sum(-1)).mean())
    out["flatness"] = float((np.exp(np.log(s).mean(-1)) / s.mean(-1)).mean())
    out["rms_dbfs"] = float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12))
    out["clipped"] = int((np.abs(x) >= 0.999).sum())
    return out


def main():
    base = Path(sys.argv[1])
    bs = {w.stem: stats(w) for w in sorted(base.glob("*.wav"))}
    for d in map(Path, sys.argv[2:]):
        rs = {w.stem: stats(w) for w in sorted(d.glob("*.wav")) if w.stem in bs}
        keys = [k for k in next(iter(bs.values())) if k not in ("clipped",)]
        diff = {k: round(float(np.mean([rs[l][k] - bs[l][k] for l in rs])), 3) for k in keys}
        flags = [k for k in diff if "-" in k and abs(diff[k]) > 3.0]
        print(json.dumps(dict(run=str(d), base=str(base), lines=len(rs), mean_diff=diff, clipped=sum(r["clipped"] for r in rs.values()), flags=flags)), flush=True)


if __name__ == "__main__":
    main()
