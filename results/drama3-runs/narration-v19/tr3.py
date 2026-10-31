import sys, glob
from faster_whisper import WhisperModel
m = WhisperModel("/opt/ep0/models/faster-whisper-small", device="cpu", compute_type="int8")
for f in sorted(glob.glob("/opt/ep0/v19/out_c/*.wav")):
    ws = [w for s in m.transcribe(f, language="en", word_timestamps=True)[0] for w in s.words]
    hit = [(w.word.strip(), round(w.start, 2), round(w.end - w.start, 2)) for w in ws if any(c in w.word for c in "RTFM-") and w.word.strip()[:1] in "RTFM"]
    print(f.split("/")[-1], hit[:5])
