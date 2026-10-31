"""Score rendered test lines for Episode 0 and append one row per line to results/drama3.csv.

Usage:
    python score.py --system p150-bfp4 --audio-dir out/p150-bfp4 --ref-voice ref_sebastian.wav [--timing out/p150-bfp4/timing.json]

The audio dir holds <line id>.wav for every line in lines.yaml. The optional timing.json maps
line id -> {"compute_s": ..., "ttfa_s": ..., "power_w": ...} as written by the generation script.

Metrics:
    wer          word error rate of faster-whisper (small.en) against the line's written form
    similarity   cosine similarity of ECAPA speaker embeddings (speechbrain) to the reference voice
    rtf          compute seconds per second of audio (lower is faster; 1.0 = real time)
"""
import argparse
import csv
import datetime
import json
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results" / "drama3.csv"
FIELDS = ["date", "system", "line", "audio_s", "wer", "similarity", "compute_s", "rtf", "ttfa_s", "power_w", "transcript"]


def load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    g = gcd(16000, sr)
    return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9$ ]+", " ", text.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="label for this run, e.g. p150-bfp4, reference-h100, chatterbox-4080")
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--ref-voice", required=True, type=Path)
    ap.add_argument("--timing", type=Path)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    import torch

    lines = yaml.safe_load((HERE / "lines.yaml").read_text(encoding="utf-8"))["lines"]
    timing = json.loads(args.timing.read_text()) if args.timing else {}
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    def embed(a: np.ndarray) -> torch.Tensor:
        return spk.encode_batch(torch.from_numpy(a)[None]).squeeze()

    ref_emb = embed(load_16k(args.ref_voice))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, FIELDS)
        if new_file:
            out.writeheader()
        for line in lines:
            path = args.audio_dir / f"{line['id']}.wav"
            if not path.exists():
                print(f"missing {path}, skipped")
                continue
            audio = load_16k(path)
            hyp = " ".join(s.text.strip() for s in asr.transcribe(audio, beam_size=5)[0])
            sim = torch.nn.functional.cosine_similarity(embed(audio), ref_emb, dim=0).item()
            t = timing.get(line["id"], {})
            audio_s = len(audio) / 16000
            row = {
                "date": datetime.date.today().isoformat(),
                "system": args.system,
                "line": line["id"],
                "audio_s": round(audio_s, 2),
                "wer": round(wer(line["written"], hyp), 4),
                "similarity": round(sim, 4),
                "compute_s": t.get("compute_s", ""),
                "rtf": round(t["compute_s"] / audio_s, 3) if "compute_s" in t else "",
                "ttfa_s": t.get("ttfa_s", ""),
                "power_w": t.get("power_w", ""),
                "transcript": hyp,
            }
            out.writerow(row)
            print(f"{line['id']:15s} wer={row['wer']:.3f} sim={row['similarity']:.3f} rtf={row['rtf']}")


if __name__ == "__main__":
    main()
