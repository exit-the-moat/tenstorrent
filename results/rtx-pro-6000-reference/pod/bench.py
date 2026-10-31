"""Drama 3 reference bench against the official SGLang-Omni server (sgl-omni serve, default settings).

    python3 bench.py <in_dir> <out_dir>

in_dir: ref.wav (reference voice), lines.json {"ref_text", "lines": [{"id", "text"}]}.
Every request: POST /v1/audio/speech, stream=true, stream_format=sse (base64 PCM deltas, then a done event whose
usage.output_tokens is the frame count), the reference sent inline as base64, max_new_tokens 600 (= the card's
max_frames). The server has no seed parameter (speech_request.py lists `seed` as unsupported), so no seed is set.
Timing is client side on localhost: t0 just before the POST, ttfa = first non-empty PCM delta, gen = done event.
"""
import base64
import json
import sys
import threading
import time
import wave
from pathlib import Path

import requests

URL = "http://127.0.0.1:30000/v1/audio/speech"
HOP = 2048  # codec samples per frame at 44.1 kHz (cookbook: frames of 2048 samples)
WARMUP = "This is a warm-up line. It is not counted in the results."

src, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.load(open(src / "lines.json"))
REF = {"data": base64.b64encode((src / "ref.wav").read_bytes()).decode(), "media_type": "audio/wav", "text": cfg["ref_text"]}


def speak(text, wav_path):
    body = {"input": text, "references": [REF], "stream": True, "stream_format": "sse", "response_format": "pcm",
            "max_new_tokens": 600}
    t0 = time.time()
    r = requests.post(URL, json=body, stream=True, timeout=600)
    r.raise_for_status()
    sr = int(r.headers.get("X-Sample-Rate", "44100"))
    pcm, t_first, first_samples, chunks, done = bytearray(), None, 0, 0, None
    for line in r.iter_lines():
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "speech.audio.delta":
            b = base64.b64decode(ev["audio"])
            if b and t_first is None:
                t_first = time.time()
                first_samples = len(b) // 2
            pcm += b
            chunks += 1
        elif ev.get("type") == "speech.audio.done":
            done = ev
        elif ev.get("type") == "error":
            raise RuntimeError(json.dumps(ev))
    t1 = time.time()
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    samples = len(pcm) // 2
    audio_s = samples / sr
    frames = (done or {}).get("usage", {}).get("output_tokens") if done else None
    gen_s = t1 - t0
    ttfa = t_first - t0
    rest_frames = (samples - first_samples) / HOP
    return {
        "text": text,
        "sample_rate": sr,
        "audio_s": round(audio_s, 3),
        "gen_s": round(gen_s, 3),
        "rtf": round(gen_s / audio_s, 4),
        "ttfa_s": round(ttfa, 4),
        "frames_usage": frames,
        "frames_from_samples": round(samples / HOP, 1),
        "ms_per_frame_whole_request": round(1e3 * gen_s / (samples / HOP), 2),
        "ms_per_frame_after_first_audio": round(1e3 * (t1 - t_first) / rest_frames, 2) if rest_frames > 0 else None,
        "chunks": chunks,
        "first_chunk_s": round(first_samples / sr, 3),
        "finish_reason": (done or {}).get("finish_reason"),
        "start_epoch": round(t0, 3),
        "first_audio_epoch": round(t_first, 3),
        "end_epoch": round(t1, 3),
    }


def sequential(name, lines):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    res = {}
    for l in lines:
        res[l["id"]] = speak(l["text"], d / f"{l['id']}.wav")
        print(name, l["id"], json.dumps(res[l["id"]]), flush=True)
    json.dump(res, open(d / "timing.json", "w"), indent=1)
    return res


def concurrent(name, lines):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    res, errs = {}, []

    def job(l):
        try:
            res[l["id"]] = speak(l["text"], d / f"{l['id']}.wav")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{l['id']}: {e}")

    t0 = time.time()
    th = [threading.Thread(target=job, args=(l,)) for l in lines]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    summ = {"streams": len(lines), "wall_s": round(wall, 3), "audio_s": round(sum(r["audio_s"] for r in res.values()), 3),
            "errors": errs, "start_epoch": round(t0, 3), "end_epoch": round(t0 + wall, 3)}
    summ["audio_s_per_wall_s"] = round(summ["audio_s"] / wall, 3)
    res["_summary"] = summ
    json.dump(res, open(d / "timing.json", "w"), indent=1)
    print(name, json.dumps(summ), flush=True)


w = speak(WARMUP, out / "warmup.wav")
json.dump(w, open(out / "warmup.json", "w"), indent=1)
print("warmup", json.dumps(w), flush=True)
sequential("seq", cfg["lines"])
sequential("seq2", cfg["lines"])
concurrent("c8", cfg["lines"])
print("BENCH_DONE", flush=True)
