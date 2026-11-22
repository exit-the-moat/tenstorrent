"""Episode 4: card side of the per-step PCC check for the lightx2v 4-step DISTILL (I2V) (same method as pcc_card.py for T2V).

Runs the I2V pipeline on a small config (480p, 17 frames) with the EP4_* config from the environment, captures at the
chosen steps the latent (16 ch), the condition (mask + VAE-encoded start image, 20 ch), timestep and expert, plus the
umT5 embeddings; then one card DiT forward (cond / uncond) per captured step on exactly those inputs.
Writes /out/pcc/<tag>/step<k>_in.pt (incl. "cond" for the CPU reference) and step<k>_card.pt.
"""
import argparse
import os

import numpy as np
import torch
import ttnn
from PIL import Image
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tensor import float32_tensor
import json

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--dump", default="0,1,2,3")
ap.add_argument("--links", type=int, default=4)
ap.add_argument("--dynamic-load", type=int, default=1)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--image", default="/out/start.png")
ap.add_argument("--prompt", default="A lone figure stands on the castle battlements watching over the glowing teal moat; wind moves his clothes, clouds drift across the golden sky. Static camera.")
a = ap.parse_args()
DUMP = [int(x) for x in a.dump.split(",")]
out = f"/out/pcc/{a.tag}"
os.makedirs(out, exist_ok=True)
CKPT = "/models/wan22-i2v-a14b"
import ep4_distill_sched as DS  # sampler: EP4_DISTILL_SAMPLER=ref (LightX2V reference, default) or tt
BOUNDARY = DS.BOUNDARY
LX_DIR = "/models/lightx2v"
os.environ.setdefault("LIGHTX2V_LOCAL_DIR", LX_DIR)
LX_FILES = ["wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors", "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"]
W, H = 832, 480

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
    checkpoint_name=CKPT, height=H, width=W, num_frames=a.frames, model_type="i2v", boundary_ratio=BOUNDARY, cfg_enabled=False,
)
pipe = WanDistillPipelineI2V(device=mesh, config=cfg, allow_download=False)
DS.install(pipe)  # LightX2V reference timesteps / Euler solver (no-op for EP4_DISTILL_SAMPLER=tt)

cap = {}
orig_enc = pipe._text_encoder.encode_cfg


def enc(*args, **kw):
    p, n = orig_enc(*args, **kw)
    cap["prompt"] = tensor.local_device_to_torch(p).float()
    cap["neg"] = None if n is None else tensor.local_device_to_torch(n).float()
    return p, n


pipe._text_encoder.encode_cfg = enc
F = (a.frames - 1) // 4 + 1
Hh, Ww = H // 8, W // 8


def unpatch(x, C, model, N):
    """(1,1,Npad,pF*pH*pW*C) -> (1,C,F,H,W); same as postprocess_spatial_output_host but with C channels."""
    pF, pH, pW = model.patch_size
    x = x.squeeze(0)[:, :N]
    x = x.reshape(1, F // pF, Hh // pH, Ww // pW, pF, pH, pW, C)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(1, C, F, Hh, Ww)


orig_step = pipe._step
steps = {}


def step(**kw):
    i = kw["step"]
    if i in DUMP:
        ts = kw["ts"]
        N = kw["latents_sequence_length"]
        lat = unpatch(tensor.local_device_to_torch(kw["permuted_latent_tt"]).float(), 16, ts.model, N)
        cond = unpatch(tensor.local_device_to_torch(kw["cond_latents"]).float(), 20, ts.model, N)
        idx = pipe.transformer_states.index(ts)
        steps[i] = dict(latent=lat.clone(), cond=cond.clone(), t=float(kw["t"]),
                        expert=["transformer", "transformer_2"][idx], idx=idx)
    return orig_step(**kw)


pipe._step = step
img = Image.open(a.image).convert("RGB")
cw = round(img.height * W / H)
img = img.crop((0, 0, cw, img.height)).resize((W, H), Image.LANCZOS)
frames = pipe(prompts=[a.prompt], image_prompt=img, num_inference_steps=a.steps, seed=a.seed, guidance_scale=1.0,
              guidance_scale_2=1.0, boundary_ratio=BOUNDARY, output_type="uint8")
np.save(f"{out}/video.npy", np.asarray(frames[0], dtype=np.uint8))

for i, s in sorted(steps.items()):
    torch.save(dict(latent=s["latent"], cond=s["cond"], t=s["t"], expert=s["expert"], prompt=cap["prompt"],
                    negative=None, ckpt=CKPT, lightx2v_dir=LX_DIR, lightx2v_file=LX_FILES[s["idx"]]),
               f"{out}/step{i}_in.pt")
    pipe._prepare_transformer(s["idx"])
    model = pipe.transformer_states[s["idx"]].model
    tt_t = float32_tensor(torch.tensor([s["t"]], dtype=torch.float32).reshape(1, 1, 1, 1), device=mesh)
    res = {}
    for k, emb in (("cond", cap["prompt"]),):  # distill: no negative-prompt pass
        emb_tt = tensor.from_torch(emb.reshape(1, 1, -1, 4096), device=mesh, dtype=ttnn.bfloat16)
        res[k] = model.forward(s["latent"], emb_tt, tt_t, y=s["cond"]).float()
    torch.save(res, f"{out}/step{i}_card.pt")
    print(f"card forward step {i} t={s['t']:.1f} {s['expert']} done", flush=True)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_DISTILL_CARD_DONE", flush=True)
