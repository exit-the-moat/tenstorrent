import sys
from faster_whisper import WhisperModel
for name in ["faster-whisper-small", "faster-whisper-base"]:
    m = WhisperModel("/opt/ep0/models/" + name, device="cpu", compute_type="int8")
    for f in sys.argv[1:]:
        print(name, f.split("/")[-1], " ".join(s.text.strip() for s in m.transcribe(f, language="en")[0]))
