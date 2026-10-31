"""port10: markdown rows from cbatch run dirs: rows10.py <dir>[=<label>] ..."""
import json
import re
import statistics as S
import sys
from pathlib import Path


def mmm(v):
    v = sorted(v)
    return f"{v[0]:.2f} / {S.median(v):.2f} / {v[-1]:.2f}"


print("| run | config | B | audio s per wall s | mean active | per-stream RTF (decode) | TTFA from admission s | ms/step bb + tower + host (draw, codec) | admission s | DRAM GB | board W mean (max) | Whisper mean (min) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    d, _, label = arg.partition("=")
    d = Path(d)
    t = json.load(open(d / "timing.json"))
    s = t["_summary"]
    j = [v for k, v in t.items() if not k.startswith("_")]
    try:
        bp = json.load(open(d / "power.txt")).get("board_power", {})
        w = f'{bp["mean"]:.0f} ({bp["max"]:.0f})'
    except Exception:
        w = "missing"
    wc = Path(str(d) + ".wcheck.log")
    m = re.search(r"mean match ([0-9.]+) min ([0-9.]+) clipped (\d+)", wc.read_text()) if wc.exists() else None
    wtxt = f"{m.group(1)} ({m.group(2)}){'' if m.group(3) == '0' else ', clipped ' + m.group(3)}" if m else "missing"
    host = f'{s["host_ms_per_step"]}'
    if "draw_ms_per_step" in s:
        host += f' ({s["draw_ms_per_step"]}, {s["codec_ms_per_step"]})'
    print(f'| {d.name} | {label} | {s["B"]} | **{s["audio_per_wall"]:.2f}** | {s["mean_active"]} | {mmm([x["rtf_decode"] for x in j])} | '
          f'{mmm([x["ttfa_from_admit_s"] for x in j])} | {s["backbone_ms_per_step"]} + {s["tower_ms_per_step"]} + {host} | '
          f'{s["admit_s_total"]:.1f} | {s.get("dram_gb")} | {w} | {wtxt} |')
