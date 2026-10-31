# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix cache vs whole-prompt prefill on the card (eager, no traces): final-norm rows of the line suffix,
first-frame candidate logits, and the first tower step's head logits (codes forced to the whole-prompt draw).

    (fast env) python -m models.demos.blackhole.drama3.tests.check_prefix_cache --lines close,fix --out x.json
"""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, log, reference_codes


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="close,fix")
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    ap.add_argument("--out", default="/opt/ep0/day4b/check_prefix.json")
    a = ap.parse_args()
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")  # these checks open the card with the GDN L1_SMALL size
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ids = a.lines.split(",")
    ref_text = Path(a.ref).with_suffix(".txt").read_text().strip()
    codes = reference_codes(a.ref, "/opt/ep0/day3")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
        s = FastSession(mesh)
        refs = [(ref_text, codes)]
        prompts = {i: s.prompt(lines[i]["text"], refs) for i in ids}
        P = s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + list(prompts.values()))
        for i in ids:
            rows = prompts[i]
            T = rows.shape[0]
            # whole prompt
            s.P = 0
            nf, lf = s._prefill_line(rows)
            hd = s._row_dev(nf[-1])
            fr_f, _, _, hl_f = s.tower.step(lf, hd, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0), collect=True)
            ttnn.deallocate(hd)
            # prefix + suffix
            s.P = P
            s.set_voice(rows)
            npf, lp = s._prefill_line(rows)
            hd = s._row_dev(npf[-1])
            fr_p, _, _, hl_p = s.tower.step(
                lp, hd, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0), collect=True, force_codes=fr_f[1:]
            )
            ttnn.deallocate(hd)
            topf, topp = lf.topk(10).indices, lp.topk(10).indices
            r = dict(
                prompt_rows=T,
                prefix_rows=P,
                suffix_normed_pcc=pcc(nf[P:], npf),
                suffix_normed_row_pcc_min=min(pcc(nf[P + k], npf[k]) for k in range(T - P)),
                cand_logits_pcc=pcc(lf, lp),
                cand_logits_max_abs=float((lf - lp).abs().max()),
                top1_same=bool(topf[0] == topp[0]),
                top10_overlap=len(set(topf.tolist()) & set(topp.tolist())),
                head_logits_pcc=[round(pcc(x, y), 5) for x, y in zip(hl_f, hl_p)],
                first_frame_full=fr_f.tolist(),
                first_frame_prefix=fr_p.tolist(),
            )
            res[i] = r
            log(f"{i}: {json.dumps(r)}")
    finally:
        ttnn.close_device(mesh)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
