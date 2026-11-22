"""Episode 4 NVIDIA reference: LTX-2.5 distilled with audio, official Lightricks ltx_pipelines DistilledPipeline
(LTX-2 repo f0f2f86, same commit as the tt port's CPU reference), bf16 weights as shipped, conv video VAE
(same decoder as the tt port), default attention (PyTorch SDPA), no FP8, no compile.

Per clip: total wall (stream_chunks + encode_media_segments, i.e. prompt encode -> mp4 on disk), and the parts:
prompt encode (Gemma 4 build + encode + free), image encode, stage 1, upsample, stage 2, video decode, audio decode.
"render" = total - prompt encode (comparable to the tt "wall per clip (render)" with a cached prompt).
Each timed part is bracketed by torch.cuda.synchronize().

--retain none : the pipeline as shipped (every block builds its model per call from the page cache and frees it).
--retain gpu  : ModelRegistry(RetentionPolicy(clean=cuda)): weights stay on the GPU between calls (like tt's resident DiT).
--retain pinned: weights kept in pinned host RAM, copied to the GPU per call.
"""
import argparse, json, os, sys, time, traceback

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/model")
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--height", type=int, required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--frames", type=int, default=121)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up clips (first job) before the counted runs")
ap.add_argument("--repeats", type=int, default=1, help="counted repeats of every job")
ap.add_argument("--retain", default="none", choices=["none", "gpu", "pinned"])
ap.add_argument("--vae", default="conv", choices=["conv", "diff"])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T_START = time.time()
TL = open(os.path.join(args.out, "timing.jsonl"), "a")


def log(*a):
    print(f"[{time.time() - T_START:8.1f}s]", *a, flush=True)


from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.placement import Placement, RetentionPolicy
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.media_io import encode_media_segments
from ltx_pipelines.utils.types import ImageConditioningInput
import ltx_pipelines.distilled as D

R = args.root
vae = (f"{R}/vae/ltx-2.5-video-vae-conv-bf16.safetensors" if args.vae == "conv"
       else f"{R}/vae/ltx-2.5-video-vae-bf16.safetensors")
mp = ModelPaths.from_split(
    transformer_path=f"{R}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path=f"{R}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path=vae,
    audio_vae_path=f"{R}/vae/ltx-2.5-audio-vae-bf16.safetensors",
    duration_head_path=f"{R}/model_patches/ltx-2.5-duration-head-bf16.safetensors",
)
if args.retain == "gpu":
    reg = ModelRegistry(placement_policy=RetentionPolicy(clean=Placement(torch.device("cuda")),
                                                         active=Placement(torch.device("cuda"))))
elif args.retain == "pinned":
    reg = ModelRegistry(placement_policy=RetentionPolicy(clean=Placement(torch.device("cpu"), pin_memory=True)))
else:
    reg = None
t = time.time()
pipe = D.DistilledPipeline(model_paths=mp,
                           spatial_upsampler_path=f"{R}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                           loras=(), device=torch.device("cuda"), registry=reg)
t_build = time.time() - t
log("pipeline object built", round(t_build, 2), "s; ancestral", pipe.use_ancestral_sampler, "retain", args.retain)

REC = []  # (name, seconds) for the current clip


def sync():
    torch.cuda.synchronize()


class Timed:
    def __init__(self, obj, name):
        object.__setattr__(self, "_o", obj)
        object.__setattr__(self, "_n", name)

    def __getattr__(self, k):
        return getattr(self._o, k)

    def __call__(self, *a, **k):
        sync(); t0 = time.time()
        out = self._o(*a, **k)
        if hasattr(out, "__next__"):
            def gen(t_acc=[time.time() - t0]):
                it = iter(out)
                while True:
                    sync(); t1 = time.time()
                    try:
                        x = next(it)
                    except StopIteration:
                        sync(); t_acc[0] += time.time() - t1
                        REC.append((self._n, t_acc[0]))
                        return
                    sync(); t_acc[0] += time.time() - t1
                    yield x
            return gen()
        sync()
        REC.append((self._n, time.time() - t0))
        return out


for attr, name in (("prompt_encoder", "prompt_encode"), ("image_conditioner", "image_encode"), ("stage", "stage"),
                   ("upsampler", "upsample"), ("video_decoder", "video_decode"), ("audio_decoder", "audio_decode")):
    setattr(pipe, attr, Timed(getattr(pipe, attr), name))

jobs = json.load(open(args.jobs))
jobdir = os.path.dirname(os.path.abspath(args.jobs))


def one(job, label, counted):
    REC.clear()
    torch.cuda.reset_peak_memory_stats()
    imgs = []
    if job.get("image"):
        imgs = [ImageConditioningInput.at(os.path.join(jobdir, job["image"]), 0, float(job.get("strength", 1.0)))]
    path = os.path.join(args.out, f"{job['name']}_s{job['seed']}_{args.height}p_{args.frames}f_{label}.mp4")
    sync(); t0 = time.time(); w0 = time.time()
    with torch.inference_mode():
        chunks, nf, tc = pipe.stream_chunks(prompt=job["prompt"], seed=int(job["seed"]), height=args.height,
                                            width=args.width, num_frames=args.frames, frame_rate=args.fps,
                                            images=imgs)
        encode_media_segments(chunks, fps=args.fps, output_path=path, video_chunks_number=None, color_space=None,
                              video_marker=None, audio_marker=None)
    sync(); total = time.time() - t0
    parts = {}
    st = 0
    for n, s in REC:
        if n == "stage":
            st += 1; n = f"stage{st}"
        parts[n] = round(parts.get(n, 0.0) + s, 3)
    pe = parts.get("prompt_encode", 0.0)
    other = total - sum(parts.values())
    row = dict(job=job["name"], seed=job["seed"], label=label, counted=counted, height=args.height, width=args.width,
               frames=nf, retain=args.retain, vae=args.vae, image=job.get("image"), total_s=round(total, 3),
               render_s=round(total - pe, 3), parts=parts, other_s=round(other, 3),
               peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
               t_wall_start=w0, t_wall_end=time.time(), mp4=os.path.basename(path))
    TL.write(json.dumps(row) + "\n"); TL.flush()
    log("CLIP", json.dumps(row))
    return row


try:
    for i in range(args.warmup):
        one(jobs[0], f"warm{i}", False)
    for r in range(args.repeats):
        for job in jobs:
            one(job, f"r{r}", True)
    log("ALL_DONE")
except torch.cuda.OutOfMemoryError as e:
    log("OOM", repr(e)[:400], "peak_alloc_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    TL.write(json.dumps({"oom": True, "frames": args.frames, "height": args.height, "retain": args.retain,
                         "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         "parts_so_far": REC[:], "err": repr(e)[:300]}) + "\n"); TL.flush()
    sys.exit(3)
except Exception:
    traceback.print_exc()
    sys.exit(2)
