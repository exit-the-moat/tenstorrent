"""Narration totals over the takes actually in the video: match each installed WAV (sha256 list on stdin, "<sha> <id>")
to a take in /opt/ep0/v20/out*/ by sha256, then sum compute_s / audio_s from that run's timing.json and the Whisper ratio
from its ana.json. Prints the per-line source and the totals."""
import glob, hashlib, json, os, statistics, sys

want = dict(l.split()[::-1] for l in sys.stdin.read().split("\n") if l.strip())  # id -> sha
have = {}
for w in glob.glob("/opt/ep0/v20/out*/*.wav"):
    have.setdefault(hashlib.sha256(open(w, "rb").read()).hexdigest(), w)
comp = aud = 0.0
ttfa, wr, src, miss = [], [], {}, []
for lid, sha in sorted(want.items()):
    w = have.get(sha)
    if not w:
        miss.append(lid)
        continue
    d, take = os.path.dirname(w), os.path.basename(w)[:-4]
    t = json.load(open(f"{d}/timing.json"))[take]
    comp += t["compute_s"]
    aud += t["audio_s"]
    ttfa.append(t["ttfa_s"])
    a = {x["take"]: x for x in json.load(open(f"{d}/ana.json"))}
    wr.append(a[take]["ratio"])
    src[lid] = f"{os.path.basename(d)}/{take}"
print(json.dumps(src, indent=0))
print(f"lines {len(src)} missing {miss}")
print(f"RTF {comp / aud:.4f} compute {comp:.1f} s audio {aud:.1f} s ttfa median {statistics.median(ttfa):.3f} "
      f"whisper mean {statistics.mean(wr):.4f} min {min(wr):.3f}")
