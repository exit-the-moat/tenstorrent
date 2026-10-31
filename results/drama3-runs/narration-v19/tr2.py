import sys
from faster_whisper import WhisperModel
m = WhisperModel("/opt/ep0/models/faster-whisper-base", device="cpu", compute_type="int8")
for f in sys.argv[1:]:
    print(f.split("/")[-1], "|", " ".join(s.text.strip() for s in m.transcribe(f, language="en")[0]))
