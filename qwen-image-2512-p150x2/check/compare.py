#!/usr/bin/env python3
"""Compare two image sets by prompt: PSNR, SSIM, CLIP image-image cosine (open_clip ViT-B-32, laion2b_s34b_b79k).
Writes compare_<label>.json into the current folder (or --json-dir) and prints a table. This made
results/compare_set_v3p_vs_rtx_base50.json (cards vs the RTX reference, same seeds and start noise).
Usage: python3 compare.py <dirA> <dirB> [label]   (matches files <prompt>-s<seed>.png in both; full 1088-row images)
Needs: pip install numpy pillow scikit-image torch open_clip_torch. Don't run it on the host during counted timings."""
import json, glob, os, sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch, open_clip

A, B = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
label = sys.argv[3] if len(sys.argv) > 3 else f"{os.path.basename(A)}_vs_{os.path.basename(B)}"
torch.set_num_threads(4)
model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()
def clip_emb(p):
    with torch.no_grad():
        return torch.nn.functional.normalize(model.encode_image(prep(Image.open(p).convert("RGB")).unsqueeze(0)), dim=-1)
rows = []
for f in sorted(glob.glob(A + "/*-s*.png")):
    name = os.path.basename(f)
    if "_1920x" in name or "_1080" in name:
        continue
    g = f"{B}/{name}"
    if not os.path.exists(g):
        continue
    a, b = np.array(Image.open(f).convert("RGB")), np.array(Image.open(g).convert("RGB"))
    assert a.shape == b.shape, (a.shape, b.shape)
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0
    s = ssim(a, b, channel_axis=2)
    c = float((clip_emb(f) * clip_emb(g)).sum())
    rows.append(dict(image=name, psnr=round(float(psnr), 2), ssim=round(float(s), 4), clip=round(c, 4)))
    print(f"{name:<24} PSNR {psnr:6.2f} dB  SSIM {s:.4f}  CLIP {c:.4f}", flush=True)
if rows:
    m = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("psnr", "ssim", "clip")}
    print(f"MEAN                     PSNR {m['psnr']:6.2f} dB  SSIM {m['ssim']:.4f}  CLIP {m['clip']:.4f}", flush=True)
else:
    m = {}
    print("no matching pairs")
json.dump(dict(a=A, b=B, label=label, rows=rows, mean=m), open(f"compare_{label}.json", "w"), indent=1)
print("DONE")
