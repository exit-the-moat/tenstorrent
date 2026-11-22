"""ltx1 I2V PCC: (1) image latent from the cards' VAE encoder vs the reference (both stages), (2) teacher-forced DiT steps
with the reference's own conditioning latent (video PCC over the non-conditioning tokens, audio), writes pcc.jsonl."""
import argparse, glob, json, os, time
import torch, ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="/work/ref/i0"); ap.add_argument("--pre", required=True)
ap.add_argument("--steps", default="s1_step0,s1_step7,s2_step0,s2_step2"); ap.add_argument("--tag", default="i2v")
a = ap.parse_args()
os.environ["LTX_TF_EXACT"] = "1"
meta = json.load(open(f"{a.ref}/meta.json")); H, W, F = meta["height"], meta["width"], meta["frames"]
def pcc(x, y):
    x, y = x.double().flatten(), y.double().flatten(); x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))
params = get_updated_device_params({"trace_region_size": 0, "l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config"); params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
pipe = LTXDistilledPipeline.create_pipeline(mesh_device=mesh, checkpoint_name="/models/ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    gemma_path=None, sp_axis=0, tp_axis=1, num_links=4, dynamic_load=False, topology=ttnn.Topology.Linear, is_fsdp=False,
    run_warmup=False, traced=False, num_frames=F, height=H, width=W, image_conditioning=True)
recs = []
ref_lat = {}
for st, (h, w) in ((1, (H // 2, W // 2)), (2, (H, W))):
    r = torch.load(f"{a.ref}/imgcond_{st - 1}.pt", weights_only=False)["latents"][0].float()
    ref_lat[st] = r
    img = torch.load(f"{a.pre}/{h}x{w}.pt").float()
    t = time.time(); lat = pipe.encode_image(img).float(); dt = time.time() - t
    recs.append({"step": f"image_latent_s{st}", "pcc": round(pcc(lat, r), 6), "shape": list(lat.shape), "ref_shape": list(r.shape), "sec": round(dt, 2)})
    logger.info(f"PCC {recs[-1]}")
ctx = {st: torch.load(f"{a.ref}/s{st}_context.pt", weights_only=False) for st in (1, 2)}
for name in a.steps.split(","):
    st = int(name[1]); d = torch.load(f"{a.ref}/{name}.pt", weights_only=False)
    h, w = (H // 2, W // 2) if st == 1 else (H, W)
    v0, a0 = pipe._denoise_no_guidance(ctx[st]["v_context"].float(), ctx[st]["a_context"].float(), num_frames=F, height=h, width=w,
        sigma_values=[d["sigma"], 0.0], seed=0, initial_video_latent=d["v_in"].float(), initial_audio_latent=d["a_in"].float(),
        image_cond_latent=ref_lat[st], image_cond_strength=1.0)
    ncond = ref_lat[st].shape[-1] * ref_lat[st].shape[-2]
    rv, ra = d["v_out"]["denoised"].float(), d["a_out"]["denoised"].float()
    recs.append({"step": name, "sigma": d["sigma"], "pcc_video_gen_tokens": round(pcc(v0[:, ncond:], rv[:, ncond:]), 6),
                 "pcc_audio": round(pcc(a0, ra), 6), "cond_tokens_equal_clean": float((v0[:, :ncond] - d["v_in"][:, :ncond].float()).abs().max())})
    logger.info(f"PCC {recs[-1]}")
with open("/out/pcc.jsonl", "a") as f:
    for r in recs: f.write(json.dumps({"tag": a.tag, "ref": a.ref, **r}) + "\n")
ttnn.close_mesh_device(mesh); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PCC_DONE")
