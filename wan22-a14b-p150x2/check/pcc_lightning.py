"""Episode 4 (wan2): card side of the per-step PCC check for T2V-A14B + Wan2.2-Lightning 4-step LoRA V1.1.

Same method as pcc_card.py / pcc_distill.py: run the pipeline on a small config (480p, 17 frames, seed 7) exactly as
gen_lightning.py does (runtime LoRA fused on the cards, LightX2V reference sampler, CFG off), capture at every step the
latent going into the DiT, timestep, expert and the umT5 embeddings; then one card DiT forward (cond only) per step on
exactly those inputs. Writes /out/pcc/<tag>/step<k>_in.pt (incl. the LoRA file for cpu_ref.py) and step<k>_card.pt.
--from-tag reuses another tag's inputs (and its CPU outputs) so a different card config is compared on the same inputs.
"""
import argparse
import glob
import os
import shutil

import numpy as np
import torch
import ttnn
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import WanPipelineRuntimeLoRA
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tensor import float32_tensor
import ep4_distill_sched as DS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--dump", default="0,1,2,3")
ap.add_argument("--links", type=int, default=4)
ap.add_argument("--dynamic-load", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--from-tag", default=None)
ap.add_argument("--no-runtime-lora", action="store_true")
ap.add_argument("--prompt", default="A golden retriever puppy runs through a field of tall grass toward the camera, ears flapping, late afternoon sun.")
a = ap.parse_args()
DUMP = [int(x) for x in a.dump.split(",")]
out = f"/out/pcc/{a.tag}"
os.makedirs(out, exist_ok=True)
CKPT = os.environ.get("EP4_CKPT", "/models/wan22-a14b")  # wan2: EP4_CKPT = host-merged checkpoint dir
LORA = "/models/lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
LORA_FILES = [LORA + "/high_noise_model.safetensors", LORA + "/low_noise_model.safetensors"]
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
    checkpoint_name=CKPT, height=H, width=W, num_frames=a.frames, boundary_ratio=DS.BOUNDARY, cfg_enabled=False,
)
pipe = WanPipelineRuntimeLoRA(device=mesh, config=cfg, run_warmup=False)
_orig_ptc = pipe.prepare_text_conditioning


def _ptc(tt_model, prompt_embeds, buffer, traced=False):
    return buffer if prompt_embeds is None else _orig_ptc(tt_model, prompt_embeds, buffer, traced)


pipe.prepare_text_conditioning = _ptc
if not a.no_runtime_lora:  # else: host-merged cache (EP4_MERGE_LORA_*, patch 10)
    pipe.register_lora("lightning_v11", high_path=LORA_FILES[0], low_path=LORA_FILES[1], scale=1.0)
    pipe.set_active_lora("lightning_v11")
DS.install(pipe)

steps, cap = {}, {}
if a.from_tag is None:
    orig_enc = pipe._text_encoder.encode_cfg

    def enc(*args, **kw):
        p, n = orig_enc(*args, **kw)
        cap["prompt"] = tensor.local_device_to_torch(p).float() if isinstance(p, ttnn.Tensor) else p.float()
        return p, n

    pipe._text_encoder.encode_cfg = enc
    F = (a.frames - 1) // 4 + 1
    Hh, Ww = H // 8, W // 8
    orig_step = pipe._step

    def step(**kw):
        i = kw["step"]
        if i in DUMP:
            ts = kw["ts"]
            x = tensor.local_device_to_torch(kw["permuted_latent_tt"]).float()
            lat = ts.model.postprocess_spatial_output_host(x, F=F, H=Hh, W=Ww, N=kw["latents_sequence_length"])
            idx = pipe.transformer_states.index(ts)
            steps[i] = dict(latent=lat.clone(), t=float(kw["t"]), expert=["transformer", "transformer_2"][idx], idx=idx)
        return orig_step(**kw)

    pipe._step = step
    frames = pipe(prompts=[a.prompt], num_inference_steps=4, seed=a.seed, guidance_scale=1.0, guidance_scale_2=1.0,
                  boundary_ratio=DS.BOUNDARY, output_type="uint8")
    np.save(f"{out}/video.npy", np.asarray(frames[0], dtype=np.uint8))
else:
    src = f"/out/pcc/{a.from_tag}"
    for f in sorted(glob.glob(f"{src}/step*_in.pt")):
        k = int(os.path.basename(f)[4:].split("_")[0])
        d = torch.load(f)
        steps[k] = dict(latent=d["latent"], t=d["t"], expert=d["expert"], idx=["transformer", "transformer_2"].index(d["expert"]))
        cap["prompt"] = d["prompt"]
        c = f.replace("_in.pt", "_cpu.pt")
        if os.path.exists(c):
            shutil.copy(c, out)

emb_tt = tensor.from_torch(cap["prompt"].reshape(1, 1, -1, 4096), device=mesh, dtype=ttnn.bfloat16)
for i, s in sorted(steps.items()):
    torch.save(dict(latent=s["latent"], t=s["t"], expert=s["expert"], prompt=cap["prompt"], negative=None, ckpt=CKPT,
                    lora_file=LORA_FILES[s["idx"]], lora_scale=1.0), f"{out}/step{i}_in.pt")
    pipe._prepare_transformer(s["idx"])
    model = pipe.transformer_states[s["idx"]].model
    tt_t = float32_tensor(torch.tensor([s["t"]], dtype=torch.float32).reshape(1, 1, 1, 1), device=mesh)
    res = {"cond": model.forward(s["latent"], emb_tt, tt_t).float()}
    torch.save(res, f"{out}/step{i}_card.pt")
    print(f"card forward step {i} t={s['t']:.1f} {s['expert']} done", flush=True)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_LIGHTNING_CARD_DONE", flush=True)
