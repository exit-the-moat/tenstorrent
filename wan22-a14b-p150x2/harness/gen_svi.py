"""Episode 4 (wan2): Stable Video Infinity 2.0 Pro on 2 x p150a (mesh 1x2), long clips by latent-continuity chaining.

tt_dit ships `experimental/pipelines/pipeline_wan_svi.py` (WanPipelineSVI) on top of WanPipelineI2VLora, which fuses the
LoRAs into the *torch* base weights on the host (the I2V fp32/bf16 base was deleted) and, in its fast "comfyui"
regime, stacks the LightX2V distill LoRA (whose files also carry .diff/.diff_b deltas that the runtime-LoRA loader
skips). So this runner ports SVI's conditioning onto what we have:
  - DiT = the lightx2v 4-step DISTILLED I2V weights (our bf8 TT cache, = D1r), LightX2V reference sampler (4 steps,
    Euler, 2 high + 2 low, CFG off);
  - SVI 2.0 Pro LoRA (vita-video-gen/svi-model rev f9bdd50, MIT, rank 128, scale 1.0) fused on the cards per expert
    with tt_dit's runtime-LoRA mixin (same fuse as Lightning T2V; NOTE the measured fuse precision loss);
  - SVI 2.0 Pro conditioning copied from WanPipelineSVI.prepare_latents: y = [anchor latent (the start image, every
    clip), last `num_motion_latent` latent frames of the previous clip, zeros]; clips joined dropping `overlap` frames.
Upstream's comfyui regime instead uses base I2V + LightX2V LoRA (0.5 high / 1.0 low) + SVI, UniPC 6 steps, CFG 1.5-2.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger
from PIL import Image
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines import events as ev
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import _LoRAPipelineMixin
from models.tt_dit.experimental.pipelines import pipeline_wan_svi as SV
import ep4_distill_sched as DS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--jobs", required=True, help="json list of {name, seed, prompt, image, crop_x?}")
ap.add_argument("--clips", type=int, default=4)
ap.add_argument("--links", type=int, default=4)
ap.add_argument("--dynamic-load", type=int, default=1)
ap.add_argument("--motion-latent", type=int, default=1)
ap.add_argument("--overlap", type=int, default=4)
ap.add_argument("--seed-stride", type=int, default=42)
ap.add_argument("--svi-scale", type=float, default=1.0)
ap.add_argument("--no-svi", action="store_true", help="control: same chaining without the SVI LoRA")
a = ap.parse_args()

W, H, FRAMES = 832, 480, 81
CKPT = "/models/wan22-i2v-a14b"
os.environ.setdefault("LIGHTX2V_LOCAL_DIR", "/models/lightx2v")
SVI = "/models/svi/svi/version-2.0/SVI_Wan2.2-I2V-A14B_{}_noise_lora_v2.0_pro.safetensors"
out = f"/out/{a.run}"
os.makedirs(out, exist_ok=True)
stamp = lambda: time.strftime("%H:%M:%S")  # noqa: E731


class SVIDistill(_LoRAPipelineMixin, WanDistillPipelineI2V):
    def __init__(self, *, device, config):
        self._lora_registry, self._active_adapter_name = {}, None  # mixin state (read during the I2V warm-up)
        self._num_motion_latent, self._num_overlap_frame = a.motion_latent, a.overlap
        self._anchor_latents_cache = None
        self._svi_anchor_image = self._svi_prev_last_latent = None
        orig = WanPipeline.__init__

        def _with_lora(self_, *args, **kw):  # WanPipelineI2V.__init__ does not forward lora_enabled
            kw.setdefault("lora_enabled", True)
            return orig(self_, *args, **kw)

        WanPipeline.__init__ = _with_lora
        try:
            WanDistillPipelineI2V.__init__(self, device=device, config=config, allow_download=False)
        finally:
            WanPipeline.__init__ = orig

    def __call__(self, *args, anchor_image=None, prev_last_latent=None, **kwargs):
        self._svi_anchor_image, self._svi_prev_last_latent = anchor_image, prev_last_latent
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._svi_anchor_image = self._svi_prev_last_latent = None

    prepare_latents = SV.WanPipelineSVI.prepare_latents  # uses super() -> rebound below


# WanPipelineSVI.prepare_latents calls zero-arg super() bound to WanPipelineSVI; re-create it for our class.
def _prepare_latents(self, batch_size, image_prompt, num_channels_latents=16, height=480, width=832, num_frames=81,
                     dtype=None, device=None, anchor_image=None, prev_last_latent=None):
    if anchor_image is None:
        anchor_image = getattr(self, "_svi_anchor_image", None)
    if prev_last_latent is None:
        prev_last_latent = getattr(self, "_svi_prev_last_latent", None)
    base = WanPipelineI2V.prepare_latents
    if anchor_image is None and prev_last_latent is None:
        return base(self, batch_size=batch_size, image_prompt=image_prompt, num_channels_latents=num_channels_latents,
                    height=height, width=width, num_frames=num_frames, dtype=dtype, device=device)
    anchor_pil = SV._as_anchor_image(anchor_image if anchor_image is not None else image_prompt)
    key = (id(anchor_pil), height, width, num_frames, num_channels_latents, dtype)
    c = self._anchor_latents_cache
    if c is not None and c[0] == key:
        latents, _ = WanPipeline.prepare_latents(self, batch_size=batch_size, num_channels_latents=num_channels_latents,
                                                 height=height, width=width, num_frames=num_frames, dtype=dtype, device=device)
        tt_y = c[1].clone()
    else:
        latents, tt_y = base(self, batch_size=batch_size, image_prompt=[ImagePrompt(image=anchor_pil, frame_pos=0)],
                             num_channels_latents=num_channels_latents, height=height, width=width,
                             num_frames=num_frames, dtype=dtype, device=device)
        self._anchor_latents_cache = (key, tt_y.clone())
    t_lat, n = tt_y.shape[2], 0
    if prev_last_latent is not None and self._num_motion_latent > 0:
        n = min(self._num_motion_latent, prev_last_latent.shape[2], t_lat - 1)
        if n > 0:
            tt_y[:, 4:, 1:1 + n, :, :] = prev_last_latent[:, :, -n:, :, :].to(tt_y.device, tt_y.dtype)
    if 1 + n < t_lat:
        tt_y[:, 4:, 1 + n:, :, :] = 0.0
    return latents, tt_y


SVIDistill.prepare_latents = _prepare_latents

params = get_updated_device_params({"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config")
params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
cfg = WanPipelineConfig.default(
    mesh_shape=mesh.shape, dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(2, 1)),
    vae_parallel_config=VaeHWParallelConfig.from_tuples(height=(2, 1), width=(1, 0)),
    encoder_parallel_config=EncoderParallelConfig.from_tuple((2, 1)),
    num_links=a.links, dynamic_load=bool(a.dynamic_load), topology=ttnn.Topology.Linear, is_fsdp=False,
    checkpoint_name=CKPT, height=H, width=W, num_frames=FRAMES, model_type="i2v", boundary_ratio=DS.BOUNDARY,
    cfg_enabled=False,
)
t1 = time.time()
pipe = SVIDistill(device=mesh, config=cfg)
if not a.no_svi:
    pipe.register_lora("svi20pro", high_path=SVI.format("high"), low_path=SVI.format("low"), scale=a.svi_scale)
    pipe.set_active_lora("svi20pro")
DS.install(pipe)
load_s = time.time() - t1
logger.info(f"ep4 {stamp()} SVI pipeline built+loaded in {load_s:.1f}s (svi={'off' if a.no_svi else a.svi_scale})")

for j in json.load(open(a.jobs)):
    img = Image.open(j["image"]).convert("RGB")
    if "crop_x" in j:
        cw = round(img.height * W / H)
        x0 = {"center": (img.width - cw) // 2, "right": img.width - cw}.get(j["crop_x"], j["crop_x"])
        img = img.crop((int(x0), 0, int(x0) + cw, img.height))
    img = img.resize((W, H), Image.LANCZOS)
    prev, clips, clip_s, marks_all = None, [], [], []
    tj = time.time()
    for k in range(a.clips):
        marks = {}

        def on_event(e, marks=marks):
            now = time.time()
            if isinstance(e, ev.SectionStart):
                marks[e.name + "_start"] = now
            elif isinstance(e, ev.SectionEnd):
                marks[e.name + "_end"] = now

        tc = time.time()
        frames, prev = pipe(prompts=[j["prompt"]], image_prompt=[ImagePrompt(image=img, frame_pos=0)],
                            seed=j["seed"] + k * a.seed_stride, num_inference_steps=4, guidance_scale=1.0,
                            guidance_scale_2=1.0, boundary_ratio=DS.BOUNDARY, output_type="pt_with_last_latent",
                            anchor_image=img, prev_last_latent=prev, on_event=on_event)
        clip_s.append(round(time.time() - tc, 2))
        sec = lambda n: round(marks[n + "_end"] - marks[n + "_start"], 2) if n + "_start" in marks and n + "_end" in marks else None  # noqa: E731
        marks_all.append({"denoise_s": sec("denoising"), "vae_s": sec("vae"), "encoder_s": sec("encoder")})
        clips.append(frames[0] if frames.ndim == 5 else frames)
        logger.info(f"ep4 {stamp()} {j['name']} clip {k+1}/{a.clips} {clip_s[-1]} s {marks_all[-1]}")
    total = time.time() - tj
    video = SV._concat_with_overlap(clips, overlap=a.overlap)
    v = video.float()
    if v.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
        v = v.permute(0, 2, 3, 1)
    if v.max() <= 1.5:
        v = v.clamp(0, 1) * 255
    arr = v.round().clamp(0, 255).to(torch.uint8).numpy()
    base = f"{out}/{j['name']}_s{j['seed']}"
    np.save(base + ".npy", arr)
    from models.tt_dit.utils.video import export_to_video

    export_to_video(arr, base + ".mp4", fps=16, crf=18, preset="medium")
    if arr.shape[0] > 241:
        export_to_video(arr[:241], base + "_241f.mp4", fps=16, crf=18, preset="medium")
    rec = dict(t=time.strftime("%F %T"), run=a.run, job=j["name"], seed=j["seed"], res="480p", w=W, h=H,
               frames=int(arr.shape[0]), steps=4, split="tp", links=a.links, dynamic_load=a.dynamic_load, quant=None,
               traced=0, mesh=[1, 2], total_s=round(total, 2), encoder_s=None,
               denoise_s=round(sum(m["denoise_s"] or 0 for m in marks_all), 2), step_s=None,
               vae_s=round(sum(m["vae_s"] or 0 for m in marks_all), 2), step_marks=[], load_s=round(load_s, 1),
               warmup_s=None, dram_mb=None, shape=list(arr.shape), mp4=True, clips=a.clips, clip_s=clip_s,
               clip_marks=marks_all, overlap=a.overlap, motion_latent=a.motion_latent, svi_scale=("host-merged " + os.environ.get("EP4_MERGE_LORA_SCALE", "1.0")) if os.environ.get("EP4_MERGE_LORA_TAG") else (None if a.no_svi else a.svi_scale),
               env={k2: v2 for k2, v2 in os.environ.items() if k2.startswith(("EP4_", "TT_DIT_"))})
    with open("/out/runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"ep4 {stamp()} JOB {j['name']} total {total:.1f}s clips {clip_s} frames {arr.shape[0]}")

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("GEN_DONE", flush=True)
