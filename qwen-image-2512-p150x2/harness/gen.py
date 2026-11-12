#!/usr/bin/env python3
"""Exit the Moat, Episode 3: full images on Tenstorrent p150a cards with Qwen-Image-2512 (tt_dit pipeline + patches/).
Output: $EP3_OUT/<tag>/<prompt>-s<seed>.png (+ the cropped copy), a .json with settings and timing, and one line in
$EP3_OUT/runs.jsonl. Started by ../run.sh inside the container (EP3_OUT=/out, prompts.json next to this file)."""
import argparse, json, os, sys, time
import ttnn
from PIL import Image
from common import SPLITS, DTYPES, open_mesh, close_mesh, dram_mb, stamp
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from models.tt_dit.pipelines.events import SectionEnd, SectionStart
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--prompts", default="moat", help="comma list from prompts.json")
ap.add_argument("--split", default="tp", choices=list(SPLITS)); ap.add_argument("--weights", default="bf16", choices=list(DTYPES))
ap.add_argument("--encoder", default="torch", choices=["torch", "device"]); ap.add_argument("--vae", default="torch", choices=["torch", "device"])
ap.add_argument("--steps", type=int, default=50); ap.add_argument("--guidance", type=float, default=4.0)
ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1088); ap.add_argument("--crop-h", type=int, default=1080)
ap.add_argument("--static-load", action="store_true", help="keep encoder and VAE weights on the cards (no dynamic load)"); ap.add_argument("--links", type=int, default=2); ap.add_argument("--seed", type=int, default=None, help="override the prompt's fixed seed")
ap.add_argument("--traced", action="store_true"); ap.add_argument("--no-alloc", action="store_true", help="skip the pipeline's 2-step empty allocation run")
ap.add_argument("--single-card", action="store_true", help="1x1 mesh (TT_VISIBLE_DEVICES selects the card): CFG in batch, no fabric")
ap.add_argument("--true-cfg", action="store_true", help="diffusers per-token norm rescale after CFG"); ap.add_argument("--trim-prompt", action="store_true", help="each CFG branch uses only its real tokens (diffusers behaviour)"); ap.add_argument("--embed-cache", default=None, help="directory for per-prompt-pair embedding cache"); ap.add_argument("--seed-offset", type=int, default=0, help="added to the prompt seed")
ap.add_argument("--queue", default="", help="claim dir: queue mode"); ap.add_argument("--queue-reps", type=int, default=4); ap.add_argument("--queue-procs", type=int, default=1); ap.add_argument("--warmup-steps", type=int, default=0)
ap.add_argument("--sdpa-chunks", default="", help="joint attention q x k chunk, e.g. 384x256"); ap.add_argument("--norm-links", type=int, default=0, help="links for the fused distributed norms (4-link norms fail to build)"); ap.add_argument("--mm-fid", default="", help="matmul fidelity override (numerics change)"); ap.add_argument("--center-k", action="store_true", help="label only: TT_DIT_QWEN25VL_CENTER_K=1 set in the environment"); ap.add_argument("--signposts", action="store_true", help="tracy signpost at every pipeline section start (profiling)"); ap.add_argument("--fuse", default="", help="lever flags, comma list (see README: up,addcmul,rope,attn_out,qkv,ff1,gelufast,psdpa,modcache,a8n,a8att,a8o)"); ap.add_argument("--seeds", default="", help="comma list of seeds, run every prompt with each (overrides --seed)")
a = ap.parse_args()
assert a.w % 16 == 0 and a.h % 16 == 0, "width and height must be multiples of 16"
CK = "qwen-image-2512"  # resolved relative to the working directory: run.sh mounts the weights at ./qwen-image-2512
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EP3_OUT", os.path.join(os.getcwd(), "out"))
all_prompts = json.load(open(os.environ.get("EP3_PROMPTS", os.path.join(HERE, "prompts.json"))))["prompts"]
outdir = f"{OUT}/{a.tag}"; os.makedirs(outdir, exist_ok=True)
S = dict(SPLITS[a.split])
if a.single_card:
    assert a.split == "tp"
    S = dict(cfg=(1, 0), sp=(1, 0), tp=(1, 0), enc=(1, 0), vae=(1, 0))
t_all = time.time()
mesh, fc = open_mesh(shape=(1, 1) if a.single_card else (1, 2), fabric=not a.single_card)
print(f"{stamp()} mesh {mesh.shape} open, split={a.split} weights={a.weights} encoder={a.encoder} vae={a.vae} steps={a.steps} links={a.links}", flush=True)
cfg = QwenImagePipelineConfig.default(
    mesh_shape=mesh.shape, topology=ttnn.Topology.Linear, num_links=a.links,
    dit_parallel_config=DiTParallelConfig.from_tuples(cfg=S["cfg"], sp=S["sp"], tp=S["tp"]),
    encoder_parallel_config=EncoderParallelConfig.from_tuple(S["enc"]),
    vae_parallel_config=VAEParallelConfig.from_tuple(S["vae"]),
    use_torch_text_encoder=a.encoder == "torch", use_torch_vae_decoder=a.vae == "torch",
    width=a.w, height=a.h, cfg_enabled=True, is_fsdp=False,
    dynamic_load_encoder=a.encoder == "device" and not a.static_load, dynamic_load_vae=a.vae == "device" and not a.static_load,
    checkpoint_name=CK, dit_weight_dtype=DTYPES[a.weights], dit_weight_dtype_name=a.weights,
    allocation_run=not a.no_alloc, cfg_rescale=a.true_cfg, trim_prompt=a.trim_prompt, embed_cache_dir=a.embed_cache)
t_init = time.time()
pipe = QwenImagePipeline(device=mesh, config=cfg)
init_s = time.time() - t_init
import ep3_levers as EL
from models.tt_dit.parallel.manager import CCLManager
assert a.center_k == (os.environ.get("TT_DIT_QWEN25VL_CENTER_K") == "1"), "--center-k label must match the env switch"
for tr in pipe.transformers:
    if a.sdpa_chunks:
        q_c, k_c = [int(v) for v in a.sdpa_chunks.split("x")]
        print(f"sdpa chunks {q_c}x{k_c} on {EL.set_sdpa(tr, q_c, k_c)} blocks", flush=True)
    if a.fuse:
        ff = a.fuse.split(chr(44))
        if "up" in ff:
            print(f"to_out/to_add_out weights bf8 -> bf16 on the card: {EL.upcast_weights(tr)} layers", flush=True)
        if "gelutanh" in ff:
            print(f"ff1 activation gelu tanh (as diffusers' gelu-approximate): {EL.set_gelu(tr, 'tanh')} layers", flush=True)
        if "gelufast" in ff:  # stock LUT GELU (approximation mode), labelled numerics option
            print(f"ff1 activation gelu fast LUT: {EL.set_gelu(tr, 'fast')} layers", flush=True)
        n_f = EL.set_fuse(tr, ff)
        print(f"fused block flags {a.fuse} on {n_f} blocks", flush=True)
    if a.mm_fid:
        print(f"matmul fidelity {a.mm_fid} on {EL.set_linear_fidelity(tr, a.mm_fid, mesh.arch())} layers", flush=True)
if a.norm_links:
    for i, tr in enumerate(pipe.transformers):
        mgr_n = CCLManager(pipe._submesh_devices[i], num_links=a.norm_links, topology=ttnn.Topology.Linear)
        print(f"norm layers on {a.norm_links} links: {EL.set_norm_ccl(tr, mgr_n)}", flush=True)
print(f"{stamp()} pipeline ready in {init_s:.0f}s; DRAM MB/chip: {dram_mb(pipe._submesh_devices[0])}", flush=True)
results = []


def run_one(name, seed, steps, record=True, extra=None):
    P = all_prompts[name]
    sections = {}; stack = {}
    def ev(e):
        if isinstance(e, SectionStart):
            if a.signposts:
                ttnn.synchronize_device(mesh); from tracy import signpost; signpost(e.name)
            stack[e.name] = time.time()
        elif isinstance(e, SectionEnd):
            sections.setdefault(e.name, 0.0)
            sections[e.name] += time.time() - stack.pop(e.name)
    t0 = time.time()
    imgs = pipe(prompts=[P["prompt"]], negative_prompts=[P.get("negative", "")], num_inference_steps=steps, cfg_scale=a.guidance, seed=seed, traced=a.traced, on_event=ev)
    if not record:
        print(f"{stamp()} warm-up {name} s{seed} {steps} steps {time.time()-t0:.1f}s", flush=True)
        return None
    im = imgs[0]
    png = f"{outdir}/{name}-s{seed}.png"
    im.save(png)
    top = (a.h - a.crop_h) // 2
    crop = im.crop((0, top, a.w, top + a.crop_h)); png_c = f"{outdir}/{name}-s{seed}_{a.w}x{a.crop_h}.png"; crop.save(png_c)
    t_end = time.time(); total = t_end - t0
    st = [sections[k] for k in sorted((k for k in sections if k.startswith("denoising_step_")), key=lambda x: int(x.rsplit("_", 1)[1]))]
    R = dict(tag=a.tag, prompt=name, seed=seed, mesh=f"{mesh.shape[0]}x{mesh.shape[1]}", split=a.split, weights=a.weights, encoder=a.encoder, vae=a.vae, links=a.links, true_cfg=a.true_cfg, trim_prompt=a.trim_prompt, embed_cache=bool(a.embed_cache), seed_offset=a.seed_offset, sdpa_chunks=a.sdpa_chunks, norm_links=a.norm_links, mm_fid=a.mm_fid, center_k=a.center_k, fuse=a.fuse,
             steps=steps, guidance=a.guidance, size=[a.w, a.h, a.crop_h], traced=a.traced,
             seconds_total=round(total, 2), seconds_per_step=round(sum(st) / max(1, len(st)), 3), step_first=round(st[0], 2) if st else None, step_last=round(st[-1], 2) if st else None,
             seconds_encoder=round(sections.get("encoder", 0), 2), seconds_denoising=round(sections.get("denoising", 0), 2), seconds_vae=round(sections.get("vae", 0), 2),
             images_per_hour=round(3600 / total, 2), pipeline_init_s=round(init_s, 1), dram_mb_per_chip=dram_mb(pipe._submesh_devices[0]), png=os.path.relpath(png, OUT), png_crop=os.path.relpath(png_c, OUT), date=time.strftime("%F %T"),
             t_start=round(t0, 3), t_end=round(t_end, 3), visible=os.environ.get("TT_VISIBLE_DEVICES", "all"), **(extra or {}))
    with open(f"{outdir}/{name}-s{seed}.json", "w") as f:
        json.dump({**R, "prompt_text": P["prompt"], "negative": P.get("negative", ""), "step_seconds": st}, f, indent=1)
    with open(f"{OUT}/runs.jsonl", "a") as f:
        f.write(json.dumps(R) + chr(10))
    print(f"{stamp()} RESULT {json.dumps(R)}", flush=True)
    results.append(R)
    return R


names = a.prompts.split(",")
if a.queue:
    # Queue mode: jobs = for rep in 0..reps-1, for prompt in order: seed = prompt seed + rep (as the RTX harness --seed-step 1).
    # Several processes may share one claim dir; each claims the next free job with an atomic mkdir.
    jobs = [(n, all_prompts[n]["seed"] + a.seed_offset + rep) for rep in range(a.queue_reps) for n in names]
    os.makedirs(a.queue, exist_ok=True)
    if a.warmup_steps:
        for n in dict.fromkeys(names):
            run_one(n, all_prompts[n]["seed"], a.warmup_steps, record=False)
    me = os.environ.get("TT_VISIBLE_DEVICES", "all")
    open(f"{a.queue}/ready_{me}", "w").write(str(time.time()))
    while len([f for f in os.listdir(a.queue) if f.startswith("ready_")]) < a.queue_procs:
        time.sleep(0.2)
    print(f"{stamp()} queue start ({len(jobs)} jobs, {a.queue_procs} process(es))", flush=True)
    for idx, (n, sd) in enumerate(jobs):
        try:
            os.mkdir(f"{a.queue}/job{idx:03d}")
        except FileExistsError:
            continue
        run_one(n, sd, a.steps, extra=dict(queue=a.queue, job=idx))
else:
    for name in names:
        if a.seeds:
            for sd in a.seeds.split(chr(44)):
                run_one(name, int(sd), a.steps)
        else:
            run_one(name, (a.seed if a.seed is not None else all_prompts[name]["seed"]) + a.seed_offset, a.steps)
close_mesh(mesh, fc)
print(f"DONE total {time.time()-t_all:.0f}s", flush=True)
