# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 3 run summary: per line timing breakdown, power, and the transcript match against the written form.

    python3 summarize_day3_run.py /opt/ep0/day3/out /opt/ep0/voice/lines.yaml
"""
import difflib
import json
import re
import sys

import yaml

out, lines_yaml = sys.argv[1], sys.argv[2]
L = {l["id"]: l for l in yaml.safe_load(open(lines_yaml))["lines"]}
S = {r["id"]: r for r in json.load(open(f"{out}/sanity.json"))}
T = json.load(open(f"{out}/timing.json"))


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


keys = ["audio_s", "compute_s", "prefill_s", "tail_s", "tower_prefill_s", "decode_backbone_s", "tower_steps_s", "codec_s"]
tot = dict.fromkeys(keys, 0.0)
frames = 0
for i, t in T.items():
    r = S[i]
    m = difflib.SequenceMatcher(None, words(L[i]["written"]), words(r["transcript"])).ratio()
    print(
        f"{i:15s} audio {t['audio_s']:5.2f}s frames {t['frames']:3d} compute {t['compute_s']:6.1f}s rtf {t['rtf']:5.1f} "
        f"ttfa {t['ttfa_s']:5.1f}s | prefill {t['prefill_s']:4.1f} tail {t['tail_s']:4.1f} tower_prefill {t['tower_prefill_s']:3.1f} "
        f"backbone_decode {t['decode_backbone_s']:6.1f} tower {t['tower_steps_s']:5.1f} codec {t['codec_s']:4.1f} | "
        f"ras {t['ras_redraws']} eos {t['stopped_by_eos']} | power {t.get('power_w')} W board {t.get('board_power_w')} W | "
        f"match_written {m:.3f}"
    )
    for k in keys:
        tot[k] += t[k]
    frames += t["frames"]
print({k: round(v, 1) for k, v in tot.items()})
print(
    f"overall rtf {tot['compute_s'] / tot['audio_s']:.2f}, {frames} frames, "
    f"backbone {1000 * tot['decode_backbone_s'] / frames:.0f} ms/frame, tower {1000 * tot['tower_steps_s'] / frames:.0f} ms/frame; "
    f"shares of compute: backbone decode {tot['decode_backbone_s'] / tot['compute_s']:.0%}, tower {tot['tower_steps_s'] / tot['compute_s']:.0%}, "
    f"card prefill+tail {(tot['prefill_s'] + tot['tail_s']) / tot['compute_s']:.0%}, codec {tot['codec_s'] / tot['compute_s']:.0%}, "
    f"tower prefill {tot['tower_prefill_s'] / tot['compute_s']:.0%}"
)
