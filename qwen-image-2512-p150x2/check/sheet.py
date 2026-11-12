import sys
from PIL import Image
"""Contact sheet: python3 sheet.py out.jpg <dirA> <dirB> ... (one row per folder, one column per test prompt)."""
out, dirs = sys.argv[1], sys.argv[2:]
P = ["keyart-s2", "text-s11", "portrait-s12", "detail-s13", "moat-s2"]
W, H = 480, 272
s = Image.new("RGB", (W * len(P), H * len(dirs)))
for r, d in enumerate(dirs):
    for c, p in enumerate(P):
        s.paste(Image.open(f"{d}/{p}.png").convert("RGB").resize((W, H), Image.LANCZOS), (c * W, r * H))
s.save(out, quality=88)
