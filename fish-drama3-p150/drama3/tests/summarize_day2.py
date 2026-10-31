# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Day 2 summary table: per configuration, per-layer PCC (42 real rows) in the residual and in the
final-norm space, and the last-position candidate logits vs the golden reference. Host only.

    python models/demos/blackhole/drama3/tests/summarize_day2.py lofi=/opt/ep0/day2/v3 hifi2=/opt/ep0/day2/hifi2all ...
"""

import json
import sys

import torch

from models.demos.blackhole.drama3.tt.weight_mapping import load_drama3_state_dict

G = "/opt/ep0/golden"


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main(specs):
    g = 1 + load_drama3_state_dict("/models/drama-3", norm=True)["norm.weight"].float()

    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g

    ref = [torch.load(f"{G}/layer_{i:02d}.pt")["hidden"] for i in range(40)]
    gm = json.load(open(f"{G}/manifest.json"))
    golden_top5 = [c["id"] for c in gm["top5_last_semantic_candidates"]]
    table = {}
    for spec in specs:
        name, d = spec.split("=")
        dev = torch.load(f"{d}/device_hidden_bf16.pt")
        res = [pcc(dev[f"layer_{i:02d}"], ref[i]) for i in range(40)]
        nrm = [pcc(norm(dev[f"layer_{i:02d}"]), norm(ref[i])) for i in range(40)]
        nlast = [pcc(norm(dev[f"layer_{i:02d}"])[-1], norm(ref[i])[-1]) for i in range(40)]
        c = json.load(open(f"{d}/chain_bf16.json"))
        table[name] = dict(
            residual_pcc_min=round(min(res), 5),
            residual_pcc_min_layer=res.index(min(res)),
            residual_pcc_mean=round(sum(res) / 40, 5),
            normed_pcc_min=round(min(nrm), 5),
            normed_pcc_min_layer=nrm.index(min(nrm)),
            normed_pcc_mean=round(sum(nrm) / 40, 5),
            normed_last_row_min=round(min(nlast), 5),
            final_norm_pcc=c["final_norm_pcc"],
            logits_pcc=c["last_logits_pcc_candidates"],
            top5=c["top5_device"],
            top1_match=c["top1_match"],
            top5_overlap=c["top5_overlap"],
        )
        print(name, json.dumps(table[name]))
    print("golden top5", golden_top5)
    json.dump(table, open("/opt/ep0/day2/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
