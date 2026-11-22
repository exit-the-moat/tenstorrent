"""Episode 4: Wan2.2-T2V-A14B on 2 x p150a (mesh 1x2). One process, a list of jobs, timing per section.

Writes per job: out/<run>/<name>_s<seed>.npy (uint8 T,H,W,3), .mp4 if imageio_ffmpeg is there, and one JSON line
in /out/runs.jsonl (encoder / denoise / per-step / vae seconds, config, DRAM).
"""
import argparse
import json
import os
import time

import numpy as np
import torch  # noqa: F401
import ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines import events as ev
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--res", default="480p", choices=["480p", "720p"])
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--split", default="tp", choices=["tp", "sp", "sp_fsdp"])
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--dynamic-load", type=int, default=1)
ap.add_argument("--quant", default=None, help="QuantConfig name, e.g. all_weights_bf8")
ap.add_argument("--jobs", default=None, help="json list of {name,prompt,seed}; default = one cat prompt")
ap.add_argument("--only", default=None, help="comma list of job names")
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--traced", type=int, default=0)
ap.add_argument("--latent-out", type=int, default=0, help="also save the final latent (.pt)")
ap.add_argument("--single-card", action="store_true")
a = ap.parse_args()

W, H = (832, 480) if a.res == "480p" else (1280, 720)
CKPT = "/models/wan22-i2v-a14b"  # Wan2.2-I2V-A14B-Diffusers rev 596658fd9ca6
import ep4_distill_sched as DS  # sampler: EP4_DISTILL_SAMPLER=ref (LightX2V reference, default) or tt
BOUNDARY = DS.BOUNDARY
os.environ.setdefault("LIGHTX2V_LOCAL_DIR", "/models/lightx2v")  # wan2.2_i2v_A14b_{high,low}_noise_lightx2v_4step
out = f"/out/{a.run}"
os.makedirs(out, exist_ok=True)


def stamp():
    return time.strftime("%H:%M:%S")


def open_mesh(shape):
    params = {"trace_region_size": 200_000_000 if a.traced else 0}
    if shape != (1, 1):
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
    params = get_updated_device_params(params)
    fc = params.pop("fabric_config", None)
    params.pop("fabric_tensix_config", None)
    if fc is not None:
        ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                               ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    return mesh, fc


def dram_mb(mesh):
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return round(v.total_bytes_allocated_per_bank * v.num_banks / 1e6)
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


shape = (1, 1) if a.single_card else (1, 2)
t0 = time.time()
mesh, fc = open_mesh(shape)
logger.info(f"ep4 {stamp()} mesh {shape} open in {time.time()-t0:.1f}s")

n = shape[1]
if a.split == "tp":
    dit_pc = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(n, 1))
    fsdp = False
elif a.split == "sp":
    dit_pc = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(n, 1), tp=(1, 0))
    fsdp = False
else:
    dit_pc = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(n, 1), tp=(1, 0))
    fsdp = True
vae_pc = VaeHWParallelConfig.from_tuples(height=(n, 1), width=(1, 0))
enc_pc = EncoderParallelConfig.from_tuple((n, 1))

cfg = WanPipelineConfig.default(
    mesh_shape=mesh.shape, dit_parallel_config=dit_pc, vae_parallel_config=vae_pc, encoder_parallel_config=enc_pc,
    num_links=a.links, dynamic_load=bool(a.dynamic_load), topology=ttnn.Topology.Linear, is_fsdp=fsdp,
    checkpoint_name=CKPT, height=H, width=W, num_frames=a.frames, model_type="i2v", boundary_ratio=BOUNDARY,
    cfg_enabled=False,  # CFG is baked into the distill: no negative-prompt pass (upstream runs it with scale 1.0 = wasted)
)
t1 = time.time()
pipe = WanDistillPipelineI2V(device=mesh, config=cfg, allow_download=False)  # base I2V for umT5/VAE/configs; DiT = lightx2v
DS.install(pipe)  # LightX2V reference timesteps / Euler solver (no-op for EP4_DISTILL_SAMPLER=tt)
load_s = time.time() - t1
logger.info(f"ep4 {stamp()} pipeline built+loaded in {load_s:.1f}s, DRAM/chip {dram_mb(mesh)} MB")
if a.quant:
    from models.tt_dit.pipelines.wan.quant_config import QuantConfig, set_quant_config

    set_quant_config(pipe, getattr(QuantConfig, a.quant)())
warm_s = None
if a.warmup:
    t2 = time.time()
    # the constructor already ran an untraced warm-up; only capture traces here if asked
    if a.traced:
        pipe(prompts=["warmup"], image_prompt=Image.new("RGB", (W, H)), num_inference_steps=2, guidance_scale=1,
             guidance_scale_2=1, traced=True)
    warm_s = time.time() - t2
    logger.info(f"ep4 {stamp()} warm-up (2 steps) {warm_s:.1f}s, DRAM/chip {dram_mb(mesh)} MB")

if a.jobs:
    jobs = json.load(open(a.jobs))
else:
    jobs = [{"name": "cats", "seed": 42,
             "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."}]
if a.only:
    keep = a.only.split(",")
    jobs = [j for j in jobs if j["name"] in keep]

_orig_decode = pipe._decode_latents
_cur = {}


def _decode_and_keep(latents, **kw):
    torch.save(latents.detach().clone(), _cur["base"] + "_latent.pt")  # final latent, before the VAE
    return _orig_decode(latents, **kw)


pipe._decode_latents = _decode_and_keep

for j in jobs:
    _cur["base"] = f"{out}/{j['name']}_s{j['seed']}"
    marks = {}
    steps = []

    def on_event(e):
        now = time.time()
        if isinstance(e, ev.SectionStart):
            marks[e.name + "_start"] = now
        elif isinstance(e, ev.SectionEnd):
            marks[e.name + "_end"] = now
        elif isinstance(e, ev.DenoiseStep):
            steps.append(now)

    tj = time.time()
    if j.get("chain_prev"):  # wan2: continue the previous clip from its last frame (already W x H)
        img = Image.fromarray(_last_frame)
        img.save(f"{out}/{j['name']}_s{j['seed']}_start.png")
    else:
        img = Image.open(j["image"]).convert("RGB")
    if "crop_x" in j and not j.get("chain_prev"):  # crop to the clip's aspect first (x offset in source pixels, "center" or "right"), no squeeze
        cw = round(img.height * W / H)
        x0 = {"center": (img.width - cw) // 2, "right": img.width - cw}.get(j["crop_x"], j["crop_x"])
        img = img.crop((int(x0), 0, int(x0) + cw, img.height))
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    res = pipe(prompts=[j["prompt"]], image_prompt=img, num_inference_steps=a.steps, seed=j["seed"],
               guidance_scale=1.0,  # distill: CFG baked in
               guidance_scale_2=1.0, boundary_ratio=BOUNDARY, output_type="pt_with_last_latent" if a.latent_out else "uint8",
               traced=bool(a.traced), on_event=on_event)
    total = time.time() - tj
    if a.latent_out:
        video, lat = res
        torch.save(lat, f"{out}/{j['name']}_s{j['seed']}_latent.pt")
        frames = (video[0].float().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8) \
            if hasattr(video[0], "permute") else np.asarray(video[0])
    else:
        frames = np.asarray(res[0], dtype=np.uint8)
    base = f"{out}/{j['name']}_s{j['seed']}"
    np.save(base + ".npy", frames)
    _last_frame = frames[-1].copy()
    mp4 = None
    try:
        from models.tt_dit.utils.video import export_to_video

        mp4 = export_to_video(frames, base + ".mp4", fps=16, crf=18, preset="medium")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"no mp4: {e}")

    def sec(k):
        s, e = marks.get(k + "_start"), marks.get(k + "_end")
        return round(e - s, 2) if s and e else None

    dn = sec("denoising")
    rec = dict(t=time.strftime("%F %T"), run=a.run, job=j["name"], seed=j["seed"], res=a.res, w=W, h=H,
               frames=a.frames, steps=a.steps, split=a.split, links=a.links, dynamic_load=a.dynamic_load,
               quant=a.quant, traced=a.traced, mesh=list(shape), total_s=round(total, 2), encoder_s=sec("encoder"),
               denoise_s=dn, step_s=round(dn / a.steps, 3) if dn else None, vae_s=sec("vae"),
               step_marks=[round(s - marks.get("denoising_start", tj), 2) for s in steps], load_s=round(load_s, 1),
               warmup_s=None if warm_s is None else round(warm_s, 1), dram_mb=dram_mb(mesh),
               shape=list(frames.shape), mp4=bool(mp4),
               env={k: v for k, v in os.environ.items() if k.startswith(("EP4_", "TT_DIT_"))})
    with open("/out/runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"ep4 {stamp()} JOB {j['name']} total {total:.1f}s enc {rec['encoder_s']} denoise {dn} vae {rec['vae_s']}")

ttnn.close_mesh_device(mesh)
if fc is not None:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("GEN_DONE", flush=True)
