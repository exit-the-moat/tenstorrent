#!/usr/bin/env python3
"""Print the README's results tables from the files in this folder (standard library only).
  python3 results/table.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "queue.jsonl"), encoding="utf-8") if l.strip()]

FINAL = [("tensor split (one model over both cards)", "full HD", "tensor_split_2cards_v3a8flq"),
         ("two copies (one whole model per card)", "full HD", "two_copies_2cards_v3a8flq"),
         ("guidance per card", "full HD", "guidance_per_card_2cards_v3a8flq"),
         ("one card alone", "full HD", "one_card_v3a8flq"),
         ("tensor split", "1024x1024", "tensor_split_2cards_v3a8flq_sq1024"),
         ("two copies", "1024x1024", "two_copies_2cards_v3a8flq_sq1024")]
START = {"tensor_split_2cards_v3a8flq": "tensor_split_2cards_v2", "two_copies_2cards_v3a8flq": "two_copies_2cards_v2",
         "guidance_per_card_2cards_v3a8flq": "guidance_per_card_2cards_v2", "one_card_v3a8flq": "one_card_v2"}

print("| setup | size | images/h, start 1 / start 2 | images per queue | s per image (one job) | encoder / denoising / VAE s | s per step |")
print("|---|---|---|---|---|---|---|")
for name, size, cfg in FINAL:
    rs = [r for r in rows if r["config"] == cfg][:2]
    if not rs:
        print(f"| {name} | {size} | missing | | | | |")
        continue
    a = rs[0]
    print(f"| {name} | {size} | **{' / '.join(f'{r['images_per_hour']:.2f}' for r in rs)}** | {' / '.join(str(r['images']) for r in rs)} | "
          f"{a['s_per_image_mean_job']} | {a['encoder_mean']} / {a['denoising_mean']} / {a['vae_mean']} | {a['step_mean']} |")

print()
print("| setup | config at the start of the work (v2) | final |")
print("|---|---|---|")
for name, size, cfg in FINAL[:4]:
    old = [r for r in rows if r["config"] == START[cfg]]
    new = [r for r in rows if r["config"] == cfg][:2]
    fmt = lambda rs: " / ".join(f"{r['images_per_hour']:.2f}" for r in rs)
    print(f"| {name} | {fmt(old)} | {fmt(new)} |")

p = json.load(open(os.path.join(HERE, "power_q9_b_r1_summary.json"), encoding="utf-8"))
print()
print(f"Power, tensor split, first counted start ({p['samples']} samples at ~1 Hz, {p['images']} images): cards "
      f"{p['card0_W']['mean']} + {p['card1_W']['mean']} W mean, {p['cards_Wh_per_image']} Wh per image on the cards, "
      f"{p['psu_Wh_per_image']} Wh per image at the PSU (PSU mean {p['psu_power_total_W']['mean']} W, whole box).")

c = json.load(open(os.path.join(HERE, "compare_set_v3p_vs_rtx_base50.json"), encoding="utf-8"))
print()
print("| prompt | PSNR dB | SSIM | CLIP |")
print("|---|---|---|---|")
for r in c["rows"]:
    print(f"| {r['image'].replace('.png', '')} | {r['psnr']} | {r['ssim']} | {r['clip']} |")
m = c["mean"]
print(f"| **mean** | **{m['psnr']}** | **{m['ssim']}** | **{m['clip']}** |")
