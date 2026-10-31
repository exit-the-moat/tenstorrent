# v19 listen-proxies: Whisper match vs the spoken text (tags stripped), duration, loudness spread, pitch range, median F0
# usage: ana.py LINES_JSON OUT_DIR [OUT_DIR...]
import glob, json, re, sys, difflib, numpy as np, soundfile as sf
from faster_whisper import WhisperModel
m = WhisperModel("/opt/ep0/models/faster-whisper-small", device="cpu", compute_type="int8")
L = {l["id"]: l for l in json.load(open(sys.argv[1]))}
PHON = {"M OW1 T": "moat", "T EH1 N S T AO2 R AH0 N T": "tenstorrent"}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def words(n):
    n = int(n)
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000: return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + words(n % 100))
    return words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words(n % 1000))
def num(m):
    a, b = m.group(1).replace(",", ""), m.group(2)
    return " " + words(a) + ("" if not b else " point " + " ".join(ONES[int(c)] for c in b)) + " "
def norm(s):
    s = re.sub(r"<\|phoneme_start\|>(.*?)<\|phoneme_end\|>", lambda m: PHON.get(m.group(1), "phon"), s)
    s = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", s).replace("%", " percent")
    s = re.sub(r"(\d[\d,]*)(?:\.(\d+))?", num, s).lower().replace("-", " ")
    w = re.sub(r"[^a-z0-9 ]", "", s).split()
    out = []  # letters read one by one (P S U) = the acronym
    for x in w:
        if len(x) == 1 and x not in ("a", "i") and out and out[-1].isalpha() and len(out[-1]) <= 3 and getattr(norm, "_l", 0): out[-1] += x
        else: out.append(x)
        norm._l = len(x) == 1 and x not in ("a", "i")
    return out
def ana(f):
    y, sr = sf.read(f); y = y if y.ndim == 1 else y.mean(1)
    n = sr // 20
    fr = np.array([np.sqrt(np.mean(y[i:i+n]**2)) for i in range(0, len(y)-n, n)])
    db = 20*np.log10(fr[fr > 1e-3]); f0 = []; w = int(sr*0.04)
    for i in range(0, len(y)-w, w):
        x = y[i:i+w]
        if np.sqrt(np.mean(x**2)) < 0.02: continue
        x = x - x.mean(); c = np.correlate(x, x, "full")[w-1:]
        lo, hi = sr//400, sr//70; k = lo + np.argmax(c[lo:hi])
        if c[k] > 0.4*c[0]: f0.append(sr/k)
    f0 = np.array(f0); st = 12*np.log2(f0/np.median(f0))
    return len(y)/sr, np.percentile(db,90)-np.percentile(db,10), np.percentile(st,90)-np.percentile(st,10), np.median(f0)
rows = []
for d in sys.argv[2:]:
    for f in sorted(glob.glob(f"{d}/*.wav")):
        name = f.split("/")[-1][:-4]; lid = re.sub(r"_s\d+$", "", name)
        if lid not in L: continue
        t = " ".join(s.text.strip() for s in m.transcribe(f, language="en")[0])
        ref = norm(L[lid]["spoken"]); hyp = norm(t)
        r = max(difflib.SequenceMatcher(None, ref, hyp).ratio(), difflib.SequenceMatcher(None, norm(L[lid]["written"]), hyp).ratio())
        dur, ls, pr, med = ana(f)
        rows.append(dict(f=f, id=lid, take=name, ratio=r, dur=dur, loud=ls, pitch=pr, f0=med, text=t))
        print(f"{name:16s} {r:.3f} {dur:4.1f}s loud {ls:4.1f}dB pitch {pr:4.1f}st f0 {med:4.0f}Hz | {t}", flush=True)
json.dump(rows, open(sys.argv[2] + "/ana.json", "w"), indent=1)
print("ANA_DONE")
