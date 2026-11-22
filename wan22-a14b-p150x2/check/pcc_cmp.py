"""Episode 4: PCC of card DiT outputs vs the CPU fp32 reference, per dumped step. Appends to /out/pcc.jsonl."""
import glob
import json
import os
import sys
import time

import torch

tag = sys.argv[1]
d = f"/out/pcc/{tag}"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


for f in sorted(glob.glob(f"{d}/step*_cpu.pt"), key=lambda x: int(os.path.basename(x)[4:].split("_")[0])):
    s = os.path.basename(f).split("_")[0]
    cpu = torch.load(f)
    card = torch.load(f"{d}/{s}_card.pt")
    inp = torch.load(f"{d}/{s}_in.pt")
    g = 4.0 if inp["expert"] == "transformer" else 3.0
    comb = lambda r: r["uncond"] + g * (r["cond"] - r["uncond"])  # noqa: E731
    has_u = "uncond" in card and "uncond" in cpu  # distill (CFG baked in): cond only
    rec = dict(t=time.strftime("%F %T"), tag=tag, step=s, timestep=inp["t"], expert=inp["expert"],
               pcc_cond=round(pcc(card["cond"], cpu["cond"]), 6),
               pcc_uncond=round(pcc(card["uncond"], cpu["uncond"]), 6) if has_u else None,
               pcc_cfg=round(pcc(comb(card), comb(cpu)), 6) if has_u else None,
               rel_cond=round(rel(card["cond"], cpu["cond"]), 4),
               rel_cfg=round(rel(comb(card), comb(cpu)), 4) if has_u else None)
    print(json.dumps(rec))
    with open("/out/pcc.jsonl", "a") as fo:
        fo.write(json.dumps(rec) + "\n")
