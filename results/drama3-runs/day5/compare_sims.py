# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 5: per-layer error of CPU FP32 sims with the device weight formats (cpu_layer_ref --rows --quant) against
the FP32 voice golden. Error = 1 - PCC of the RMS-normalised residual without the massive channel 1269 (pitfall 10),
over all rows; plus final norm, candidate logits and top-1 on the 24 teacher rows.

    python compare_sims.py /opt/ep0/day3/golden_voice /opt/ep0/day5/sim_qwen36 /opt/ep0/day5/sim_bfp8 [--out x.json]
"""
import argparse
import json
from pathlib import Path

import torch

MASSIVE = 1269
T = 718


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def normed(h):
    h = h.float()
    h = h / h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    keep = torch.ones(h.shape[-1], dtype=torch.bool)
    keep[MASSIVE] = False
    return h[:, keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("sims", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = Path(a.golden)
    res = {}
    gl = [normed(torch.load(g / f"layer_{i:02d}.pt")["hidden"]) for i in range(40)]
    gfn = torch.load(g / "final_norm.pt")["hidden"].float()
    gcl = torch.load(g / "cand_logits.pt")["logits"].float()
    for s in a.sims:
        s = Path(s)
        err = []
        for i in range(40):
            h = normed(torch.load(s / f"layer_{i:02d}.pt")["hidden"])
            err.append(1 - pcc(h, gl[i]))
        fn = torch.load(s / "final_norm.pt")["hidden"].float()
        cl = torch.load(s / "cand_logits.pt")["logits"].float()
        teach = range(T - 1, cl.shape[0])
        res[s.name] = dict(
            layer_err=[round(e, 7) for e in err],
            layer_err_step=[round(err[0], 7)] + [round(err[i] - err[i - 1], 7) for i in range(1, 40)],
            final_norm_pcc=round(pcc(fn, gfn), 6),
            final_norm_pcc_teacher_min=round(min(pcc(fn[r], gfn[r]) for r in teach), 6),
            cand_logits_pcc_teacher_mean=round(sum(pcc(cl[r], gcl[r]) for r in teach) / len(teach), 6),
            top1_teacher=f"{sum(int(cl[r].argmax()) == int(gcl[r].argmax()) for r in teach)}/{len(teach)}",
            top5_overlap_teacher_mean=round(
                sum(len(set(torch.topk(cl[r], 5).indices.tolist()) & set(torch.topk(gcl[r], 5).indices.tolist())) for r in teach)
                / len(teach),
                3,
            ),
        )
        print(s.name, json.dumps({k: v for k, v in res[s.name].items() if not k.startswith("layer")}), flush=True)
    names = [Path(s).name for s in a.sims]
    if len(names) == 2:
        e4, e8 = (res[n]["layer_err_step"] for n in names)
        gain = [(i, e4[i] - e8[i]) for i in range(40)]
        gain.sort(key=lambda x: -x[1])
        res["gate_up_sensitivity_ranked"] = [(i, round(d, 7)) for i, d in gain]
        print("most sensitive layers (BFP4 - BFP8 step error):", [i for i, _ in gain[:16]], flush=True)
        print("layer err", names[0], [round(x, 5) for x in res[names[0]]["layer_err"]], flush=True)
        print("layer err", names[1], [round(x, 5) for x in res[names[1]]["layer_err"]], flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
