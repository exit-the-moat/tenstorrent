"""port7: speaker similarity (resemblyzer GE2E embedding cosine) of run dirs vs the voice reference ref_v2 and vs a
base dir's take of the same line. python spk.py <base dir> <run dir> ... -> one JSON line per dir."""
import json
import sys
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

enc = VoiceEncoder("cpu")
emb = lambda p: enc.embed_utterance(preprocess_wav(Path(p)))  # noqa: E731  (unit-norm)
ref = emb("/opt/ep0/voice/ref_v2.wav")
base = Path(sys.argv[1])
be = {w.stem: emb(w) for w in sorted(base.glob("*.wav"))}
for d in [base] + [Path(x) for x in sys.argv[2:]]:
    rows = {}
    for w in sorted(d.glob("*.wav")):
        e = emb(w)
        rows[w.stem] = dict(vs_ref=float(e @ ref), vs_base=float(e @ be[w.stem]) if w.stem in be else None)
    vr = [r["vs_ref"] for r in rows.values()]
    vb = [r["vs_base"] for r in rows.values() if r["vs_base"] is not None]
    out = dict(run=str(d), n=len(rows), vs_ref_mean=round(float(np.mean(vr)), 4), vs_ref_min=round(float(np.min(vr)), 4),
               vs_base_mean=round(float(np.mean(vb)), 4) if vb else None, vs_base_min=round(float(np.min(vb)), 4) if vb else None)
    print(json.dumps(out), flush=True)
    json.dump(dict(summary=out, lines=rows), open(Path("/opt/ep0/port7") / f"spk_{d.name}.json", "w"), indent=1)
