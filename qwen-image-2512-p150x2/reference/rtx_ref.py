"""Episode 3: RTX 4080 SUPER 16 GB reference for Qwen-Image-2512 (diffusers, CPU offload, labelled so on screen).

Same prompts, seeds, size, steps, guidance and the SAME start noise as the card harness (../harness/gen.py):
torch.manual_seed(seed) + randn([1, 16, H//8, W//8]) on the CPU, packed like the pipeline packs it. The 41 GB bf16
transformer does not fit into 16 GB of VRAM, so its blocks are streamed from system RAM (group offloading).

Run (see README.md, "Quality"): python rtx_ref.py --tag base50 --prompts keyart,text,portrait,detail,moat --out DIR
Writes per image: <out>/<tag>/<prompt>-s<seed>.png, a 1920x1080 crop, a .json with settings and timing; and one line per
image in <out>/rtx_runs.jsonl. Timing is split into encoder / denoising / VAE (the offloaded RTX speed is not comparable
with a GPU that holds the whole model; we used these images for quality only).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from diffusers import QwenImagePipeline  # noqa: E402
from diffusers.hooks import apply_group_offloading  # noqa: E402


def pack(lat):
    """The pipeline's _pack_latents for a [1, 16, H, W] noise tensor."""
    b, c, h, w = lat.shape
    x = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="keyart")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1088)
    ap.add_argument("--crop-h", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=None, help="override the prompt's fixed seed")
    ap.add_argument("--repeat", type=int, default=1, help="generate the prompt list this many times (first image is warm-up for the page cache)")
    ap.add_argument("--seed-step", type=int, default=0, help="seed = base + rep * step, for a queue with different seeds per pass")
    ap.add_argument("--blocks", type=int, default=1, help="transformer blocks per offload group (use_stream needs 1)")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512", help="HF id or a local diffusers folder")
    ap.add_argument("--revision", default="25468b98e3276ca6700de15c6628e51b7de54a26", help="the HF revision we used")
    ap.add_argument("--prompts-file", default=str(HERE.parent / "harness" / "prompts.json"))
    ap.add_argument("--out", default="rtx_out")
    a = ap.parse_args()
    assert a.w % 16 == 0 and a.h % 16 == 0

    prompts = json.loads(Path(a.prompts_file).read_text(encoding="utf-8"))["prompts"]
    data = Path(a.out)
    out = data / a.tag
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dev = torch.device("cuda")
    rev = None if os.path.isdir(a.model) else a.revision
    pipe = QwenImagePipeline.from_pretrained(a.model, revision=rev, torch_dtype=torch.bfloat16)
    apply_group_offloading(pipe.text_encoder, onload_device=dev, offload_device=torch.device("cpu"), offload_type="leaf_level", low_cpu_mem_usage=True)
    pipe.transformer.enable_group_offload(onload_device=dev, offload_device=torch.device("cpu"), offload_type="block_level",
                                          num_blocks_per_group=a.blocks, use_stream=not a.no_stream, non_blocking=True, low_cpu_mem_usage=True)
    pipe.vae.to(dev)
    pipe._execution_device_override = dev
    load_s = time.time() - t0
    print(f"pipeline loaded in {load_s:.0f}s", flush=True)

    sections = {}
    def timed(name, fn):
        def w(*x, **k):
            s = time.time()
            r = fn(*x, **k)
            torch.cuda.synchronize()
            sections[name] = sections.get(name, 0.0) + time.time() - s
            return r
        return w
    pipe.encode_prompt = timed("encoder", pipe.encode_prompt)
    pipe.vae.decode = timed("vae", pipe.vae.decode)

    t_queue = time.time()
    count = 0
    for rep in range(a.repeat):
        for name in a.prompts.split(","):
            P = prompts[name]
            seed = (a.seed if a.seed is not None else P["seed"]) + rep * a.seed_step
            sections.clear()
            torch.manual_seed(seed)
            noise = pack(torch.randn([1, 16, a.h // 8, a.w // 8])).to(dev, torch.bfloat16)
            torch.cuda.reset_peak_memory_stats()
            marks = []
            def cb(_p, i, _t, kw):
                torch.cuda.synchronize()
                marks.append(time.time())
                return kw
            t1 = time.time()
            result = pipe(prompt=P["prompt"], negative_prompt=P.get("negative", ""), true_cfg_scale=a.guidance, width=a.w, height=a.h,
                          num_inference_steps=a.steps, latents=noise, callback_on_step_end=cb)
            torch.cuda.synchronize()
            total = time.time() - t1
            im = result.images[0]
            steps = [marks[i] - marks[i - 1] for i in range(1, len(marks))]
            stem = f"{name}-s{seed}" + (f"-r{rep}" if a.repeat > 1 else "")
            png = out / f"{stem}.png"
            im.save(png)
            top = (a.h - a.crop_h) // 2
            crop = out / f"{stem}_{a.w}x{a.crop_h}.png"
            im.crop((0, top, a.w, top + a.crop_h)).save(crop)
            denoise = total - sections.get("encoder", 0.0) - sections.get("vae", 0.0)
            R = dict(tag=a.tag, prompt=name, seed=seed, rep=rep, device="RTX 4080 SUPER 16 GB, CPU offload (block streaming)", weights="bf16", steps=a.steps,
                     guidance=a.guidance, size=[a.w, a.h, a.crop_h], seconds_total=round(total, 2),
                     seconds_per_step=round(sum(steps) / max(1, len(steps)), 3), step_first=round(steps[0], 2) if steps else None,
                     step_last=round(steps[-1], 2) if steps else None, seconds_encoder=round(sections.get("encoder", 0.0), 2),
                     seconds_denoising=round(denoise, 2), seconds_vae=round(sections.get("vae", 0.0), 2), images_per_hour=round(3600 / total, 2),
                     peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), pipeline_load_s=round(load_s, 1),
                     png=str(png.relative_to(data)), png_crop=str(crop.relative_to(data)), date=time.strftime("%F %T"),
                     torch=torch.__version__, blocks_per_group=a.blocks, stream=not a.no_stream)
            (out / f"{stem}.json").write_text(json.dumps({**R, "prompt_text": P["prompt"], "negative": P.get("negative", ""), "step_seconds": steps}, indent=1), encoding="utf-8")
            with open(data / "rtx_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(R) + "\n")
            print("RESULT " + json.dumps(R), flush=True)
            count += 1
    wall = time.time() - t_queue
    Q = dict(tag=a.tag, images=count, wall_s=round(wall, 1), images_per_hour=round(count * 3600 / wall, 2), steps=a.steps, device="RTX 4080 SUPER 16 GB, CPU offload (block streaming)", date=time.strftime("%F %T"))
    with open(data / "rtx_queue.jsonl", "a", encoding="utf-8") as f:
        print(json.dumps(Q), file=f)
    print("QUEUE " + json.dumps(Q), flush=True)
    print(f"DONE total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
