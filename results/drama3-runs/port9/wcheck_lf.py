"""port9: wcheck.py with the reference lines from a JSON lines file (id, written, spoken): LINES_FILE=<json>."""
import difflib, json, os, re, sys
from pathlib import Path
from models.demos.blackhole.drama3.host import transcribe as T
from faster_whisper import WhisperModel
LINES = {l["id"]: l for l in json.load(open(os.environ["LINES_FILE"]))}
words = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()
m = WhisperModel(T.MODEL, device="cpu", compute_type="int8", cpu_threads=8)
for d in map(Path, sys.argv[1:]):
    res = []
    for w in sorted(d.glob("*.wav")):
        lid = w.stem.split("_r")[0] if w.stem not in LINES else w.stem
        r = T.check(w, m, LINES[lid]["written"])
        r["match_written"] = round(difflib.SequenceMatcher(None, words(LINES[lid]["written"]), words(r["transcript"])).ratio(), 3)
        res.append(dict(id=w.stem, match=r["match_written"], clipped=r["clipped_samples"], transcript=r["transcript"]))
    json.dump(res, open(d / "sanity.json", "w"), indent=1)
    ms = [x["match"] for x in res]
    print(d.name, "n", len(ms), "mean match %.3f" % (sum(ms) / len(ms)), "min %.3f" % min(ms), "clipped", sum(x["clipped"] for x in res), flush=True)
