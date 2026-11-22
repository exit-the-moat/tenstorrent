"""Episode 4: contact sheets and clip comparison.

  python sheet.py <run>              -> out/<run>/_sheet.jpg: one row per clip, 8 frames each (labelled)
  python sheet.py --cmp A.npy B.npy  -> mean / min per-frame PSNR (dB) of B vs A, plus a 2-row side-by-side jpg
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def frames8(v):
    return [v[i] for i in np.linspace(0, len(v) - 1, 8).astype(int)]


def row(v, w=320):
    h = int(v.shape[1] * w / v.shape[2])
    tiles = [Image.fromarray(f).resize((w, h), Image.LANCZOS) for f in frames8(v)]
    r = Image.new("RGB", (w * 8, h))
    for k, t in enumerate(tiles):
        r.paste(t, (k * w, 0))
    return r


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 10, 18], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 255))
    return img


def psnr(a, b):
    m = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


if sys.argv[1] == "--cmp":
    A, B = np.load(sys.argv[2]), np.load(sys.argv[3])
    p = [psnr(x, y) for x, y in zip(A, B)]
    rec = dict(a=sys.argv[2], b=sys.argv[3], psnr_mean=round(float(np.mean(p)), 2), psnr_min=round(float(np.min(p)), 2))
    print(json.dumps(rec))
    ra, rb = label(row(A), os.path.basename(sys.argv[2])), label(row(B), os.path.basename(sys.argv[3]))
    s = Image.new("RGB", (ra.width, ra.height * 2))
    s.paste(ra, (0, 0))
    s.paste(rb, (0, ra.height))
    s.save(sys.argv[3].replace(".npy", "_vs.jpg"), quality=88)
    with open("/out/compare.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
else:
    d = f"/out/{sys.argv[1]}"
    files = sorted(glob.glob(f"{d}/*.npy"))
    rows = [label(row(np.load(f)), os.path.basename(f)[:-4]) for f in files]
    s = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        s.paste(r, (0, y))
        y += r.height
    s.save(f"{d}/_sheet.jpg", quality=85)
    print(f"{d}/_sheet.jpg", len(rows))
