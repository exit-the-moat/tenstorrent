"""port5 Track A: compare a run dir with a baseline (frames + wav sample identity, seam location) and print the
TTFA breakdown from timing.json.  python3 cmp_ttfa.py <run dir> [<base dir>=/opt/ep0/day4b/out_v2] [--json out]"""
import json, sys, wave
from pathlib import Path
import numpy as np
import torch

run = Path(sys.argv[1]); base = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("/opt/ep0/day4b/out_v2")
FL = 2048  # samples per frame at 44.1 kHz (codec frame_length), checked below against wav length

def wav(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64), w.getframerate()

t = json.load(open(run / "timing.json"))
tb = json.load(open(base / "timing.json")) if (base / "timing.json").exists() else {}
out = {}
for lid, r in t.items():
    o = {k: r.get(k) for k in ("ttfa_s", "voice_prefill_s", "prefill_s", "prefill_backbone_s", "prefill_tower_s", "first_frame_s",
                                "first_window", "first_window_pushed_s", "first_z_taken_s", "first_card_decode_s",
                                "backbone_ms_per_frame", "tower_ms_per_frame", "rtf", "frames")}
    o["base_ttfa_s"] = tb.get(lid, {}).get("ttfa_s")
    fb = base / f"{lid}.frames.pt"
    if fb.exists():
        a, b = torch.load(run / f"{lid}.frames.pt"), torch.load(fb)
        o["frames_identical"] = bool(a.shape == b.shape and torch.equal(a, b))
        x, sr = wav(run / f"{lid}.wav"); y, _ = wav(base / f"{lid}.wav")
        o["wav_identical"] = bool(x.shape == y.shape and np.array_equal(x, y))
        if x.shape == y.shape and not o["wav_identical"]:
            fl = len(x) // a.shape[0]
            d = np.nonzero(x != y)[0]
            o["frame_len"] = fl
            o["diff_first_sample"], o["diff_last_sample"] = int(d[0]), int(d[-1])
            o["diff_last_frame"] = int(d[-1]) // fl
            seg = slice(0, (int(d[-1]) // fl + 1) * fl)
            err = x[seg] - y[seg]
            o["seam_snr_db"] = round(10 * np.log10((y[seg] ** 2).sum() / max((err ** 2).sum(), 1e-9)), 1)
            o["max_abs_diff"] = int(np.abs(err).max())
    out[lid] = o
    print(lid, json.dumps(o))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
