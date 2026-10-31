# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize generate.py --golden check files (voice prompt, mixed embeddings): prefill per-layer PCC, prompt-tail
and teacher-forced decode rows (min layer PCC, final norm, candidate logits, top-1, top-5 overlap).

    python summarize_day3_check.py lofi=/opt/ep0/day3/check_lofi/close.check.json hifi2=...
"""
import json
import sys


def stats(xs):
    return f"min {min(xs):.5f} mean {sum(xs) / len(xs):.5f}"


for spec in sys.argv[1:]:
    name, path = spec.split("=")
    c = json.load(open(path))
    pl = c["prefill_layers"]
    print(f"== {name}")
    print(f"prefill (704 rows): per-layer PCC {stats(pl)} (min at layer {pl.index(min(pl))}); final norm {c['prefill_final_norm']}")
    for label, rows in (("prompt tail (decode)", c["tail_rows"]), ("teacher frames (decode)", c["decode_steps"])):
        rows = [r for r in rows if r.get("final_norm_pcc") is not None]
        print(
            f"{label} ({len(rows)} rows): layer PCC {stats([r['layer_pcc_min'] for r in rows])}; "
            f"final norm {stats([r['final_norm_pcc'] for r in rows])}; logits {stats([r['logits_pcc'] for r in rows])}; "
            f"top1 {sum(r['top1_match'] for r in rows)}/{len(rows)}; top5 overlap mean {sum(r['top5_overlap'] for r in rows) / len(rows):.2f}"
        )
    last = c["tail_rows"][-1]
    print(f"last prompt row (first frame's logits): top1 match {last['top1_match']}, top5 overlap {last['top5_overlap']}, logits PCC {last['logits_pcc']}")
