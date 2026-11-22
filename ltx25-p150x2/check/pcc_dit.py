"""ltx1: teacher-forced per-step PCC of the LTX-2.5 DiT on 1x2 vs the CPU reference dumps (ltx_pipelines, bf16).
For each reference denoiser call: same input latents + sigma + contexts -> one transformer pass on the cards ->
x0 = x - sigma*v -> PCC vs the reference denoised x0 (video and audio). Writes /out/pcc.jsonl.
"""
import argparse, glob, json, os, time
import torch, ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/work/ref/r0")
ap.add_argument("--tag", default="bf8")
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--steps", default="all", help="comma list like s1_step0,s2_step1 or all")
a = ap.parse_args()
os.environ["LTX_TF_EXACT"] = "1"
meta = json.load(open(f"{a.ref}/meta.json"))
H, W, F = meta["height"], meta["width"], meta["frames"]

def pcc(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))

params = get_updated_device_params({"trace_region_size": 0, "l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config"); params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
pipe = LTXDistilledPipeline.create_pipeline(
    mesh_device=mesh, checkpoint_name="/models/ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    gemma_path=None, sp_axis=0, tp_axis=1, num_links=a.links, dynamic_load=False, topology=ttnn.Topology.Linear,
    is_fsdp=False, run_warmup=False, traced=False, num_frames=F, height=H, width=W)
files = sorted(glob.glob(f"{a.ref}/s[12]_step*.pt"))
if a.steps != "all":
    keep = set(a.steps.split(","))
    files = [f for f in files if os.path.basename(f)[:-3] in keep]
ctx = {st: torch.load(f"{a.ref}/s{st}_context.pt", weights_only=False) for st in (1, 2) if os.path.exists(f"{a.ref}/s{st}_context.pt")}
for f in files:
    name = os.path.basename(f)[:-3]
    st = int(name[1])
    d = torch.load(f, weights_only=False)
    h, w = (H // 2, W // 2) if st == 1 else (H, W)
    t = time.time()
    v0, a0 = pipe._denoise_no_guidance(
        ctx[st]["v_context"].float(), ctx[st]["a_context"].float(), num_frames=F, height=h, width=w,
        sigma_values=[d["sigma"], 0.0], seed=0,
        initial_video_latent=d["v_in"].float(), initial_audio_latent=d["a_in"].float())
    dt = time.time() - t
    rv, ra = d["v_out"]["denoised"].float(), d["a_out"]["denoised"].float()
    rec = {"tag": a.tag, "ref": a.ref, "step": name, "sigma": d["sigma"], "pcc_video": round(pcc(v0, rv), 6),
           "pcc_audio": round(pcc(a0, ra), 6), "rel_l2_video": round(float((v0 - rv).norm() / rv.norm()), 5),
           "sec": round(dt, 2), "cfg": {k: v for k, v in os.environ.items() if k.startswith("LTX_")}}
    logger.info(f"PCC {rec}")
    with open("/out/pcc.jsonl", "a") as fo:
        fo.write(json.dumps(rec) + "\n")
if os.path.exists(f"{a.ref}/upsampler_in.pt") and a.steps in ("all", "ups") or "ups" in a.steps:
    from models.tt_dit.models.vae.vae_ltx import upsample_latent
    ui = torch.load(f"{a.ref}/upsampler_in.pt", weights_only=False)["args"][0].float()
    uo = torch.load(f"{a.ref}/upsampler_out.pt", weights_only=False).float()
    pipe._prepare_upsampler()
    up = upsample_latent(pipe.upsampler, ui, *pipe._vae_per_channel_stats())
    torch.save(up, f"/work/ups_{a.tag}.pt")
    rec = {"tag": a.tag, "ref": a.ref, "step": "upsampler", "pcc": round(pcc(up, uo), 6), "shape": list(up.shape)}
    logger.info(f"PCC {rec}")
    open("/out/pcc.jsonl", "a").write(json.dumps(rec) + chr(10))
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_DONE")
