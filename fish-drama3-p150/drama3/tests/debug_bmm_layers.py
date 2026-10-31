# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b debug: whole-prompt prefill with the grouped-MoE bmm configs on vs off, per-layer PCC of the outputs;
then, at the first bad layer, the grouped MoE alone on that layer's (config-off) input, on vs off."""
import json
import os
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if not torch.isfinite(a).all():
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    from models.demos.blackhole.qwen36.tt.moe.prefill_grouped import grouped_prefill_forward

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ref = str(VOICE / "ref_v2.wav")
    refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        os.environ["DRAMA3_CARD_CODEC"] = "0"
        s = FastSession(mesh)
        rows = s.prompt(lines["fix"]["text"], refs)
        E = s.embedder(rows)
        outs = {}
        for flag in ("0", "1"):
            os.environ["QWEN36_GROUPED_BMM_CONFIG"] = flag
            s.bb.reset()
            _, _, per_layer = s.bb.prefill(E, collect=True)
            outs[flag] = per_layer
        bad = None
        for i, (a, b) in enumerate(zip(outs["0"], outs["1"])):
            p = pcc(a, b)
            print(json.dumps(dict(layer=i, pcc=round(p, 6), finite=bool(torch.isfinite(b).all()))), flush=True)
            if bad is None and not p > 0.999:
                bad = i
        print("first bad layer", bad, flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
