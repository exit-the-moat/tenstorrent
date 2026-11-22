"""Episode 4 (ltx1): LTX-2.5 distilled text-to-audio+video on 2 x p150a (mesh 1x2). One process, a list of jobs.

Prompt embeddings come from tt_dit's embedding disk cache (TT_DIT_CACHE_DIR/ltx-embeddings/<md5>.device.pt), written
by an earlier host-CPU encoder, or by the Gemma 4 port on the cards (--gemma 1, used for every counted run).
Writes per job: /out/<run>/<name>_s<seed>.mp4 and one JSON line in /out/runs.jsonl (stage timings).
"""
import argparse
import json
import os
import time

import torch
import ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params

from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--height", type=int, default=448)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--frames", type=int, default=121)
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--dynamic-load", type=int, default=0)
ap.add_argument("--jobs", default=None, help="json list of {name,prompt,seed}")
ap.add_argument("--only", default=None)
ap.add_argument("--repeat", type=int, default=1, help="run the job list this many times (warm repeats)")
ap.add_argument("--gemma", type=int, default=0, help="1 = Gemma 4 text encoder on the cards (else embedding cache only)")
ap.add_argument("--save-pt", type=int, default=0)
ap.add_argument("--image", default=None, help="I2V start image: preprocessed dir (prep_img.py) or file; per-job key image overrides")
ap.add_argument("--image-strength", type=float, default=1.0)
ap.add_argument("--warmup", type=int, default=1, help="1 = first job once untimed (compile) before the list")
a = ap.parse_args()

if a.frames < 121 and os.environ.get("LTX_CONV3D_BORROW") == "1":  # ltx1: borrowed blockings hung twice at 25 frames
    os.environ["LTX_CONV3D_BORROW"] = "0"
    logger.warning("LTX_CONV3D_BORROW disabled for clips shorter than 121 frames")
CKPT = "/models/ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
out = f"/out/{a.run}"
os.makedirs(out, exist_ok=True)


def open_mesh(shape):
    params = {"trace_region_size": 0, "l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
    params = get_updated_device_params(params)
    fc = params.pop("fabric_config", None)
    params.pop("fabric_tensix_config", None)
    ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                           ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)


def dram_mb(mesh):
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return round(v.total_bytes_allocated_per_bank * v.num_banks / 1e6)
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


jobs = json.load(open(a.jobs)) if a.jobs else [json.load(open("/ltx/harness/prompts8.json"))[0]]
if a.only:
    keep = set(a.only.split(","))
    jobs = [j for j in jobs if j["name"] in keep]

t0 = time.time()
mesh = open_mesh((1, 2))
logger.info(f"ltx1 mesh (1,2) open in {time.time()-t0:.1f}s")
t0 = time.time()
pipe = LTXDistilledPipeline.create_pipeline(
    mesh_device=mesh,
    checkpoint_name=CKPT,
    gemma_path="/models/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" if a.gemma else None,
    sp_axis=0,
    tp_axis=1,
    num_links=a.links,
    dynamic_load=bool(a.dynamic_load),
    topology=ttnn.Topology.Linear,
    is_fsdp=False,
    run_warmup=False,
    traced=False,
    num_frames=a.frames,
    height=a.height,
    width=a.width,
    image_conditioning=bool(a.image) or any("image" in j for j in jobs),
)
t_load = time.time() - t0
logger.info(f"ltx1 pipeline ready in {t_load:.1f}s, DRAM {dram_mb(mesh)} MB")

env_keys = sorted(k for k in os.environ if k.startswith(("LTX_", "EP4_", "TT_DIT")))
cfg = {k: os.environ[k] for k in env_keys}


def run_segments(j, path, imgs):
    """ltx1 I2V chaining: segment n+1 starts from the last decoded frame of segment n (strength 1 pins it);
    the duplicated first frame (and its 1/24 s of audio) of every later segment is dropped at the joint."""
    import torch.nn.functional as Fn
    from models.tt_dit.utils.video import Audio, export_video_audio
    vids, wavs, seg_t, sr = [], [], [], None
    img = imgs[0][0] if imgs else None
    for n in range(int(j["segments"])):
        t0 = time.time()
        segimgs = [(img, 0, imgs[0][2] if imgs else 1.0)] if img else None
        vid, aud = pipe.generate(j["prompt"], output_path=None, output_type="float", images=segimgs,
                                 num_frames=a.frames, height=a.height, width=a.width, seed=j["seed"] + n)
        seg_t.append(round(time.time() - t0, 2))
        vid = vid.float()
        w = aud.waveform.float(); sr = aud.sampling_rate
        if n > 0:
            vid = vid[:, :, 1:]
            w = w[..., int(round(sr / 24)):]
        vids.append(vid); wavs.append(w)
        last = vids[-1][:, :, -1:].clone()  # (1,3,1,H,W) in [-1,1]
        d = f"{out}/_seg_{j['name']}_{n}"
        os.makedirs(d, exist_ok=True)
        torch.save(last, f"{d}/{a.height}x{a.width}.pt")
        half = Fn.interpolate(last[:, :, 0], size=(a.height // 2, a.width // 2), mode="bilinear", antialias=True,
                              align_corners=False).unsqueeze(2)
        torch.save(half, f"{d}/{a.height // 2}x{a.width // 2}.pt")
        img = d
    video = torch.cat(vids, dim=2)
    wav = torch.cat(wavs, dim=-1)
    crop = (lambda v: v[..., 4:1084, :]) if a.height == 1088 else (lambda v: v)  # 1920x1088 -> 1920x1080 masters
    export_video_audio(crop(video), path, fps=24, audio=Audio(waveform=wav, sampling_rate=sr))
    if j.get("first_out"):  # the first segment alone = the 5 s clip (same seed and start image)
        export_video_audio(crop(vids[0]), j["first_out"], fps=24, audio=Audio(waveform=wavs[0], sampling_rate=sr))
    if a.save_pt:
        torch.save({"video": video.half(), "seg_frames": [v.shape[2] for v in vids]}, path.replace(".mp4", ".pt"))
    return seg_t


def run_segments_overlap(j, path, imgs):
    """ltx1 O1: official-style chunk carry: chunk n+1 is conditioned on the last `overlap` pixel frames of chunk n
    (their latent frames at both stage resolutions, strength 1); the re-rendered overlap is cross-faded (video and audio)."""
    from models.tt_dit.utils.video import Audio, export_video_audio
    ov = int(j.get("overlap", 25)); fc = (ov - 1) // 8 + 1
    vids, wavs, seg_t, ov_psnr, sr = [], [], [], [], None
    last = None
    for n in range(int(j["segments"])):
        t0 = time.time()
        if n > 0:
            pipe._ltx_carry = {"s1": last["s1"][:, :, -fc:].clone(), "s2": last["s2"][:, :, -fc:].clone()}
        try:
            vid, aud = pipe.generate(j["prompt"], output_path=None, output_type="float", images=imgs if n == 0 else None,
                                     num_frames=a.frames, height=a.height, width=a.width, seed=j["seed"] + n)
        finally:
            pipe._ltx_carry = None
        seg_t.append(round(time.time() - t0, 2))
        last = {k: v.clone() for k, v in pipe._ltx_last.items()}
        vid = vid.float(); w = aud.waveform.float(); sr = aud.sampling_rate
        if n > 0:
            prev, prevw = vids[-1], wavs[-1]
            mse = ((prev[:, :, -ov:] - vid[:, :, :ov]).clamp(-2, 2) * 0.5).pow(2).mean()
            ov_psnr.append(round(float(-10 * torch.log10(mse)), 2))
            wt = torch.linspace(0, 1, ov).view(1, 1, ov, 1, 1)
            vids[-1] = torch.cat([prev[:, :, :-ov], prev[:, :, -ov:] * (1 - wt) + vid[:, :, :ov] * wt], dim=2)
            vid = vid[:, :, ov:]
            na = int(round(sr * ov / 24))
            aw = torch.linspace(0, 1, na)
            wavs[-1] = torch.cat([prevw[..., :-na], prevw[..., -na:] * (1 - aw) + w[..., :na] * aw], dim=-1)
            w = w[..., na:]
        vids.append(vid); wavs.append(w)
    video = torch.cat(vids, dim=2); wav = torch.cat(wavs, dim=-1)
    crop = (lambda v: v[..., 4:1084, :]) if a.height == 1088 else (lambda v: v)
    export_video_audio(crop(video), path, fps=24, audio=Audio(waveform=wav, sampling_rate=sr))
    logger.info(f"ltx1 overlap chain: frames {video.shape[2]} overlap PSNR (prev tail vs re-rendered head) {ov_psnr}")
    return seg_t, ov_psnr


def run_job(j, tag, counted):
    if j.get("single_out") and os.path.exists(j["single_out"]) and os.environ.get("LTX_SKIP_DONE") == "1":
        logger.info(f"ltx2 skip {j['name']}: {j['single_out']} exists")
        return
    path = f"{out}/{j['name']}_s{j['seed']}{tag}.mp4"
    img = j.get("image", a.image)
    imgs = [(img, 0, float(j.get("strength", a.image_strength)))] if img else None
    t = time.time()
    seg_t = None
    ov_psnr = None
    pipe._ltx_audio_frozen = torch.load(j["audio"]).float() if j.get("audio") else None  # ltx1 A1 lip-sync
    if j.get("segments") and j.get("overlap"):
        seg_t, ov_psnr = run_segments_overlap(j, path, imgs)
    elif j.get("segments"):
        seg_t = run_segments(j, path, imgs)
    elif j.get("single_out"):  # ltx2: one single pass, 1088 -> 1080 crop, written to the job's own path
        from models.tt_dit.utils.video import Audio, export_video_audio
        vid, aud = pipe.generate(j["prompt"], output_path=None, output_type="float", images=imgs, num_frames=a.frames,
                                 height=a.height, width=a.width, seed=j["seed"])
        crop = (lambda v: v[..., 4:1084, :]) if a.height == 1088 else (lambda v: v)
        path = j["single_out"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        export_video_audio(crop(vid.float()), path, fps=24, audio=Audio(waveform=aud.waveform.float(), sampling_rate=aud.sampling_rate))
    elif a.save_pt:
        vid, aud = pipe.generate(j["prompt"], output_path=None, output_type="float", images=imgs, num_frames=a.frames, height=a.height, width=a.width, seed=j["seed"])
        torch.save({"video": vid.float(), "audio": aud}, path.replace(".mp4", ".pt"))
        from models.tt_dit.utils.video import export_video_audio
        export_video_audio(vid, path, fps=24, audio=aud)
    else:
        pipe.generate(j["prompt"], output_path=path, images=imgs, num_frames=a.frames, height=a.height, width=a.width, seed=j["seed"])
    wall = time.time() - t
    rec = {
        "run": a.run, "name": j["name"], "seed": j["seed"], "counted": counted, "tag": tag,
        "height": a.height, "width": a.width, "frames": a.frames, "links": a.links, "dynamic_load": a.dynamic_load,
        "wall_s": round(wall, 2), "timings": [(k, round(v, 2)) for k, v in getattr(pipe, "last_timings", [])],
        "dram_mb": dram_mb(mesh), "enc_s": enc_times.get(j["name"]), "seg_s": seg_t, "segments": j.get("segments"), "overlap": j.get("overlap"), "overlap_psnr": ov_psnr, "audio": j.get("audio"), "image": img, "strength": (imgs[0][2] if imgs else None), "start": time.strftime("%F %T", time.localtime(t)), "cfg": cfg, "out": path,
    }
    with open("/out/runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"ltx1 JOB {j['name']} {tag} wall {wall:.1f}s timings {rec['timings']}")


enc_times = {}
if a.gemma:
    # ltx1: encode every prompt up front on the cards (timed), keep the embeddings in the disk cache, then free Gemma
    # so DiT + VAE get the memory. Per-clip encode time is reported from here (runs.jsonl field enc_s).
    os.environ.pop("LTX_NO_EMBED_CACHE", None)
    pipe.gemma_encoder_pair._encoder_trace = False
    _t = time.time(); pipe.gemma_encoder_pair.ensure_loaded(); t_gload = time.time() - _t
    _t = time.time(); pipe.encode_prompts(["warmup"], use_cache=False); t_gwarm = time.time() - _t
    logger.info(f"ltx1 gemma load {t_gload:.2f}s first encode (compile) {t_gwarm:.2f}s")
    for j in jobs:
        t = time.time(); res = pipe.encode_prompts([j["prompt"]], use_cache=False)
        enc_times[j["name"]] = round(time.time() - t, 3)
        torch.save(res, pipe._device_embed_cache_path([j["prompt"]]))
    logger.info(f"ltx1 device encode times {enc_times}")
    _p = pipe.gemma_encoder_pair
    _t = time.time()
    for _m in (_p.gemma_encoder, _p.feature_extractor, _p.video_connector, _p.audio_connector):
        if _m is not None and _m.is_loaded():
            _m.deallocate_weights()
    logger.info(f"ltx1 gemma free {time.time()-_t:.2f}s")
    if os.environ.get("LTX_GEMMA_RELOAD_TEST") == "1":  # second load = the per-prompt reload cost (warm cache)
        _t = time.time(); _p.ensure_loaded(); t2 = time.time() - _t
        _t = time.time(); pipe.encode_prompts([jobs[0]["prompt"] + " "], use_cache=False); t3 = time.time() - _t
        for _m in (_p.gemma_encoder, _p.feature_extractor, _p.video_connector, _p.audio_connector):
            if _m is not None and _m.is_loaded():
                _m.deallocate_weights()
        logger.info(f"ltx1 gemma reload {t2:.2f}s encode {t3:.2f}s")
if a.warmup:
    run_job(jobs[0], "_warm", False)
for r in range(a.repeat):
    for j in jobs:
        run_job(j, f"_r{r}", True)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
logger.info("ltx1 all done")
