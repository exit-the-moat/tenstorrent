"""Episode 4: card side of the per-step PCC check.

Runs the pipeline on a small config, captures at chosen steps the latent going into the DiT, the timestep and the
expert, plus the umT5 prompt embeddings; then runs one card DiT forward (cond and uncond, no CFG mix) per captured
step on exactly those inputs. Writes /out/pcc/<tag>/step<k>_in.pt (CPU reference input) and step<k>_card.pt.
"""
import argparse
import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tensor import float32_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--dump", default="0,20,35,39")
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--split", default="tp", choices=["tp", "sp", "sp_fsdp"])
ap.add_argument("--quant", default=None)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--from-tag", default=None, help="skip the pipeline run: reuse <tag> step inputs (CPU ref exists)")
ap.add_argument("--prompt", default="A golden retriever puppy runs through a field of tall grass toward the camera, ears flapping, late afternoon sun.")
a = ap.parse_args()
DUMP = [int(x) for x in a.dump.split(",")]
out = f"/out/pcc/{a.tag}"
os.makedirs(out, exist_ok=True)

params = get_updated_device_params({"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config")
params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)

if a.split == "tp":
    dit_pc, fsdp = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(2, 1)), False
else:
    dit_pc, fsdp = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(2, 1), tp=(1, 0)), a.split == "sp_fsdp"
cfg = WanPipelineConfig.default(
    mesh_shape=mesh.shape, dit_parallel_config=dit_pc,
    vae_parallel_config=VaeHWParallelConfig.from_tuples(height=(2, 1), width=(1, 0)),
    encoder_parallel_config=EncoderParallelConfig.from_tuple((2, 1)),
    num_links=a.links, dynamic_load=True, topology=ttnn.Topology.Linear, is_fsdp=fsdp,
    checkpoint_name="/models/wan22-a14b", height=480, width=832, num_frames=a.frames,
)
pipe = WanPipeline(device=mesh, config=cfg, run_warmup=False)
if a.quant:
    from models.tt_dit.pipelines.wan.quant_config import QuantConfig, set_quant_config

    set_quant_config(pipe, getattr(QuantConfig, a.quant)())

if a.from_tag is None:
    cap = {}
    orig_enc = pipe._text_encoder.encode_cfg


    def enc(*args, **kw):
        p, n = orig_enc(*args, **kw)
        cap["prompt_tt"], cap["neg_tt"] = p, n
        cap["prompt"] = tensor.local_device_to_torch(p).float() if isinstance(p, ttnn.Tensor) else p.float()
        cap["neg"] = tensor.local_device_to_torch(n).float() if isinstance(n, ttnn.Tensor) else n.float()
        return p, n


    pipe._text_encoder.encode_cfg = enc
    F = (a.frames - 1) // 4 + 1
    Hh, Ww = 480 // 8, 832 // 8
    orig_step = pipe._step
    steps = {}


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
    t0 = time.time()
    frames = pipe(prompts=[a.prompt], num_inference_steps=a.steps, seed=a.seed, guidance_scale=4.0, guidance_scale_2=3.0,
                  output_type="uint8")
    logger.info(f"ep4 pcc run {time.time()-t0:.0f}s")
    np.save(f"{out}/video.npy", np.asarray(frames[0], dtype=np.uint8))
    print("prompt emb", tuple(cap["prompt"].shape), flush=True)
else:
    import glob, shutil
    steps, cap = {}, {}
    src = f"/out/pcc/{a.from_tag}"
    for f in sorted(glob.glob(f"{src}/step*_in.pt")):
        k = int(os.path.basename(f)[4:].split("_")[0])
        d = torch.load(f)
        steps[k] = dict(latent=d["latent"], t=d["t"], expert=d["expert"], idx=["transformer", "transformer_2"].index(d["expert"]))
        cap["prompt"], cap["neg"] = d["prompt"], d["negative"]
        shutil.copy(f.replace("_in.pt", "_cpu.pt"), out) if os.path.exists(f.replace("_in.pt", "_cpu.pt")) else None
    cap["prompt_tt"] = tensor.from_torch(cap["prompt"].reshape(1, 1, -1, 4096), device=mesh, dtype=ttnn.bfloat16)
    cap["neg_tt"] = tensor.from_torch(cap["neg"].reshape(1, 1, -1, 4096), device=mesh, dtype=ttnn.bfloat16)

for i, s in sorted(steps.items()):
    torch.save(dict(latent=s["latent"], t=s["t"], expert=s["expert"], prompt=cap["prompt"], negative=cap["neg"]),
               f"{out}/step{i}_in.pt")
    pipe._prepare_transformer(s["idx"])
    model = pipe.transformer_states[s["idx"]].model
    tt_t = float32_tensor(torch.tensor([s["t"]], dtype=torch.float32).reshape(1, 1, 1, 1), device=mesh)
    res = {}
    for k, emb in (("cond", cap["prompt_tt"]), ("uncond", cap["neg_tt"])):
        res[k] = model.forward(s["latent"], emb, tt_t).float()
    torch.save(res, f"{out}/step{i}_card.pt")
    print(f"card forward step {i} t={s['t']:.1f} {s['expert']} done", flush=True)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_CARD_DONE", flush=True)
