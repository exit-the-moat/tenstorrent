# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does the last-position error of the chained device run come from? (host only)

Applies the reference final norm (FP32) to every layer's device and golden residual, so each layer is
compared in the space the LM head sees, where the few massive residual channels (absmax ~24k) no longer
dominate the PCC.

    python models/demos/blackhole/drama3/tests/analyze_chain.py /opt/ep0/day2/v3/device_hidden_bf16.pt
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


def main(path):
    dev = torch.load(path)
    w = load_drama3_state_dict("/models/drama-3", norm=True)["norm.weight"].float()
    g = 1 + w

    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g

    ref39 = torch.load(f"{G}/layer_39.pt")["hidden"]
    top = ref39[-1].abs().topk(8)
    print("largest residual channels (last row, layer 39):", top.indices.tolist())
    print("their |value|:", [round(v, 1) for v in top.values.tolist()])
    print("their norm gain 1+w:", [round(v, 4) for v in g[top.indices].tolist()])
    print("norm gain range %.4f .. %.4f" % (g.min(), g.max()))

    rows = []
    for i in range(40):
        d, r = dev[f"layer_{i:02d}"], torch.load(f"{G}/layer_{i:02d}.pt")["hidden"]
        mask = torch.ones(d.shape[-1], dtype=torch.bool)
        mask[top.indices] = False
        rows.append(
            dict(
                layer=i,
                residual_last=round(pcc(d[-1], r[-1]), 5),
                residual_last_no_outliers=round(pcc(d[-1, mask], r[-1, mask]), 5),
                normed_last=round(pcc(norm(d)[-1], norm(r)[-1]), 5),
                normed_all=round(pcc(norm(d), norm(r)), 5),
            )
        )
        print(json.dumps(rows[-1]))
    dn = dev["final_norm"]
    print("device final norm vs host norm of device residual (last row): %.6f" % pcc(dn[-1], norm(dev["layer_39"])[-1]))


if __name__ == "__main__":
    main(sys.argv[1])
