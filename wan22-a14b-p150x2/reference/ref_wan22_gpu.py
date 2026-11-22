"""Episode 4: NVIDIA reference for Wan2.2-T2V-A14B (stock diffusers), self-contained for a fresh GPU pod.

Same job as the Tenstorrent runs: Wan-AI/Wan2.2-T2V-A14B-Diffusers rev 5be7df9619b5 (bf16 weights in memory), UniPC
flow_shift 12, 40 steps, CFG 4.0 (high-noise expert) / 3.0 (low-noise), boundary 0.875, the card pipeline's default
negative prompt, 512 text tokens, 81 frames, 16 fps, start noise = CPU randn after torch.manual_seed(seed) (passed as
`latents`, identical to the card pipeline). The 8 harness prompts + seeds are embedded below (same as prompts8.json).

Run on the pod (1 GPU, 80-96 GB: A100 80GB / H100 80GB / RTX PRO 6000 96GB):
    bash setup_pod.sh                      # pip installs + pinned download (~126 GB, fp32 experts)
    python ref_wan22_gpu.py --res 480p     # then --res 720p
    (--only motion,face  for a subset; --offload if the GPU has < 80 GB; --warmup 0 to skip the 2-step warm-up)
Writes ./out_<res>/<name>_s<seed>.mp4 + .npy (uint8 T,H,W,3), ./results_gpu.csv (one row per clip: total, encoder,
denoise, per step, VAE, peak memory, GPU name, versions) and timing.jsonl.

Timing split (all with torch.cuda.synchronize): encoder = pipe.encode_prompt (umT5, prompt + negative), denoise = the
pipeline call with precomputed embeddings and output_type="latent" (per-step marks via callback), VAE = vae.decode of
the final latent + postprocess. Total = sum of the three (+ host glue).
"""
import argparse
import csv
import json
import os
import platform
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video

REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REV = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: E501, RUF001
PROMPTS = [
    ("motion", 101, "A red sports car drifts around a wet racetrack corner at dusk, spraying water, motion blur, the camera tracking alongside at low height."),
    ("face", 102, "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, slowly turning his head toward the camera and smiling, soft morning light, shallow depth of field."),
    ("text", 103, "A neon sign reading \"OPEN LATE\" flickers on above the door of a small noodle shop on a rainy street at night, reflections in the puddles."),
    ("crowd", 104, "A busy pedestrian crossing in Tokyo seen from above, hundreds of people with umbrellas crossing in all directions, light rain."),
    ("pan", 105, "Slow camera pan from left to right across a mountain lake at sunrise, pine forest, mist on the water, snow-capped peaks in the background."),
    ("hands", 106, "Hands of a potter shaping a clay bowl on a spinning wheel, wet clay, close-up, warm workshop light."),
    ("animal", 107, "A golden retriever puppy runs through a field of tall grass toward the camera, ears flapping, late afternoon sun."),
    ("moat", 108, "A dark stone castle built of circuit boards stands behind a glowing teal moat at night; a small figure climbs a golden ladder out of the moat while the sun rises behind the hills, cinematic, slow push-in."),
]

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="480p", choices=["480p", "720p"])
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--only", default=None, help="comma list of prompt names")
ap.add_argument("--ckpt", default=os.environ.get("WAN_DIR", "./wan22-a14b"), help="local snapshot dir (setup_pod.sh)")
ap.add_argument("--offload", action="store_true", help="model CPU offload (GPUs with < 80 GB)")
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--prompts", default=None, help="json list of {name,seed,prompt}; default = the 8 embedded prompts")
ap.add_argument("--out", default=None, help="output dir (default ./out_<res>); timing.jsonl + results_gpu.csv go here too")
ap.add_argument("--rev", default=REV, help="only used if --ckpt is a hub id; a local snapshot is already pinned")
a = ap.parse_args()
W, H = (832, 480) if a.res == "480p" else (1280, 720)
out = a.out or f"./out_{a.res}"
os.makedirs(out, exist_ok=True)
if a.prompts:
    PROMPTS = [(j["name"], j["seed"], j["prompt"]) for j in json.load(open(a.prompts))]
REVKW = {} if os.path.isdir(a.ckpt) else {"revision": a.rev}
dev = "cuda"


def sync():
    torch.cuda.synchronize()


vae = AutoencoderKLWan.from_pretrained(a.ckpt, subfolder="vae", torch_dtype=torch.float32, **REVKW)
pipe = WanPipeline.from_pretrained(a.ckpt, vae=vae, torch_dtype=torch.bfloat16, **REVKW)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(a.ckpt, subfolder="scheduler", flow_shift=12.0, **REVKW)
if a.offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to(dev)
F = (a.frames - 1) // 4 + 1


def start_noise(seed):
    torch.manual_seed(seed)  # identical to the Tenstorrent pipeline (CPU randn, float32)
    return torch.randn((1, 16, F, H // 8, W // 8), dtype=torch.float32)


def run(prompt, seed):
    sync()
    t0 = time.time()
    pe, ne = pipe.encode_prompt(prompt=prompt, negative_prompt=NEG, do_classifier_free_guidance=True,
                                max_sequence_length=512, device=dev, dtype=torch.bfloat16)
    sync()
    t1 = time.time()
    marks = []

    def cb(p, i, t, kw):
        sync()
        marks.append(time.time())
        return kw

    lat = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=H, width=W, num_frames=a.frames,
               num_inference_steps=a.steps, guidance_scale=4.0, guidance_scale_2=3.0,
               latents=start_noise(seed).to(dev), output_type="latent", callback_on_step_end=cb).frames
    sync()
    t2 = time.time()
    # VAE decode exactly as WanPipeline does for output_type="np"
    lat = lat.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
    video = pipe.vae.decode(lat / std + mean, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    sync()
    t3 = time.time()
    return frames, dict(encoder_s=t1 - t0, denoise_s=t2 - t1, vae_s=t3 - t2, total_s=t3 - t0, marks=marks, t1=t1)


if a.warmup:
    run("warmup", 0)

jobs = [p for p in PROMPTS if not a.only or p[0] in a.only.split(",")]
gpu = torch.cuda.get_device_name(0)
fields = ["time", "gpu", "res", "job", "seed", "frames", "steps", "total_s", "encoder_s", "denoise_s", "s_per_step", "vae_s",
          "video_min_per_hour", "max_mem_gb", "offload", "torch", "diffusers", "cuda", "host"]
RES = os.path.join(out, "results_gpu.csv")
new = not os.path.exists(RES)
for name, seed, prompt in jobs:
    torch.cuda.reset_peak_memory_stats()
    frames, t = run(prompt, seed)
    u8 = (np.asarray(frames) * 255).round().clip(0, 255).astype(np.uint8)
    np.save(f"{out}/{name}_s{seed}.npy", u8)
    export_to_video(list(frames), f"{out}/{name}_s{seed}.mp4", fps=16)
    import diffusers
    row = dict(time=time.strftime("%F %T"), gpu=gpu, res=a.res, job=name, seed=seed, frames=a.frames, steps=a.steps,
               total_s=round(t["total_s"], 2), encoder_s=round(t["encoder_s"], 3), denoise_s=round(t["denoise_s"], 2),
               s_per_step=round(t["denoise_s"] / a.steps, 3), vae_s=round(t["vae_s"], 2),
               video_min_per_hour=round(60 * (a.frames / 16) / t["total_s"], 3),
               max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), offload=a.offload, torch=torch.__version__,
               diffusers=diffusers.__version__, cuda=torch.version.cuda, host=platform.node())
    with open(RES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
            new = False
        w.writerow(row)
    with open(os.path.join(out, "timing.jsonl"), "a") as f:
        f.write(json.dumps({**row, "step_s": row["s_per_step"], "step_marks": [round(m - t["t1"], 3) for m in t["marks"]]}) + "\n")
    print(json.dumps(row), flush=True)
print("REF_DONE")
