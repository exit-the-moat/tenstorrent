# LTX-2.5 distilled CPU reference with dumps (ltx1, Episode 4).
# Runs the official ltx_pipelines DistilledPipeline on the host CPU (bf16 weights, as shipped) and dumps
# prompt embeddings, every denoiser call (inputs/outputs, both stages), upsampler in/out, decoder in/out,
# audio, and the final mp4 into OUT.
import os, sys, time, json, argparse, dataclasses, logging
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/work/ref/r0")
ap.add_argument("--prompt", default=None)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--height", type=int, default=448)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--frames", type=int, default=25)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--vae", default="conv", choices=["conv", "diff"])
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--hidden", type=int, default=0)
ap.add_argument("--image", default=None)
ap.add_argument("--strength", type=float, default=1.0)
args = ap.parse_args()
torch.set_num_threads(args.threads)
logging.basicConfig(level=logging.INFO)
os.makedirs(args.out, exist_ok=True)
R = os.environ.get("LTX_WEIGHTS", "/models/ltx25")  # the Lightricks/LTX-2.5 download
PROMPT = args.prompt or json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness", "prompts8.json")))[0]["prompt"]

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)

def cpu(obj, depth=0):
    """Recursively copy tensors (and dataclass/tuple/list/dict of tensors) to CPU clones."""
    if depth > 6:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").clone()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {"__type__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            try:
                d[f.name] = cpu(getattr(obj, f.name), depth + 1)
            except Exception as e:
                d[f.name] = f"<err {e}>"
        return d
    if isinstance(obj, (list, tuple)):
        return [cpu(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: cpu(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # generic object: tensor attributes only
    d = {"__type__": type(obj).__name__}
    for k, v in getattr(obj, "__dict__", {}).items():
        if isinstance(v, torch.Tensor) or dataclasses.is_dataclass(v):
            d[k] = cpu(v, depth + 1)
    return d

def save(name, obj):
    torch.save(obj, os.path.join(args.out, name + ".pt"))
    log("saved", name)

from ltx_pipelines.utils.model_paths import ModelPaths
import ltx_pipelines.distilled as D
import ltx_core.text_encoders.gemma.feature_extractor as FE

# --- hooks ---------------------------------------------------------------
_fe_calls = [0]
_orig_fe = FE.FeatureExtractorV2.forward
def fe_forward(self, hidden_states, attention_mask, padding_side="left"):
    out = _orig_fe(self, hidden_states, attention_mask, padding_side)
    if args.hidden and _fe_calls[0] == 0:
        hs = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        save("gemma_hidden", {"hidden": hs.detach().to(torch.bfloat16).cpu(), "mask": attention_mask.cpu()})
    _fe_calls[0] += 1
    save(f"feature_out_{_fe_calls[0]-1}", cpu(out))
    return out
FE.FeatureExtractorV2.forward = fe_forward

stage_ctr = [0]
_orig_loop = D.euler_ancestral_denoising_loop
def loop(sigmas, video_state, audio_state, transformer, denoiser, v_context=None, a_context=None, **kw):
    st = stage_ctr[0] + 1
    stage_ctr[0] += 1
    save(f"s{st}_context", {"v_context": cpu(v_context), "a_context": cpu(a_context), "sigmas": cpu(sigmas),
                            "kw": {k: v for k, v in kw.items() if isinstance(v, (int, float, str))}})
    save(f"s{st}_state_in", {"video": cpu(video_state), "audio": cpu(audio_state)})
    def den(tr, vs, as_, sig, idx, vc, ac):
        t = time.time()
        vr, ar = denoiser(tr, vs, as_, sig, idx, vc, ac)
        log(f"stage {st} step {idx} sigma {float(sig[idx]):.6f} denoiser {time.time()-t:.1f}s")
        save(f"s{st}_step{idx}", {"sigma": float(sig[idx]), "v_in": cpu(vs.latent if vs is not None else None),
                                  "a_in": cpu(as_.latent if as_ is not None else None),
                                  "v_out": cpu(vr), "a_out": cpu(ar)})
        return vr, ar
    res = _orig_loop(sigmas, video_state, audio_state, transformer, den, v_context, a_context, **kw)
    save(f"s{st}_state_out", cpu(res))
    return res
D.euler_ancestral_denoising_loop = loop

vae = f"{R}/vae/ltx-2.5-video-vae-conv-bf16.safetensors" if args.vae == "conv" else f"{R}/vae/ltx-2.5-video-vae-bf16.safetensors"
mp = ModelPaths.from_split(
    transformer_path=f"{R}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path=f"{R}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path=vae,
    audio_vae_path=f"{R}/vae/ltx-2.5-audio-vae-bf16.safetensors",
    duration_head_path=f"{R}/model_patches/ltx-2.5-duration-head-bf16.safetensors",
)
log("building pipeline")
pipe = D.DistilledPipeline(model_paths=mp,
                           spatial_upsampler_path=f"{R}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                           loras=(), device=torch.device("cpu"))
log("use_ancestral_sampler", pipe.use_ancestral_sampler)
# fp32 for the conv parts (torch CPU bf16 conv3d is single-threaded and slow); transformer stays bf16 as shipped
pipe.upsampler._dtype = torch.float32
_orig_ic = D.image_conditionings_for_chunk
_icn = [0]
def ic(chunk, **kw):
    items = _orig_ic(chunk, **kw)
    save(f"imgcond_{_icn[0]}", {"latents": [cpu(it.latent) for it in items], "strengths": [it.strength for it in items]})
    _icn[0] += 1
    return items
D.image_conditionings_for_chunk = ic

# wrap upsampler / decoders (block call objects)
def wrap_block(obj, name):
    orig = obj.__call__
    class W:
        pass
    def call(*a, **k):
        save(f"{name}_in", {"args": cpu(list(a)), "kwargs": cpu(k)})
        t = time.time()
        if name == "upsampler":  # fp32 upsampler model: feed fp32, return the pipeline dtype
            out = orig(*[x.float() if isinstance(x, torch.Tensor) else x for x in a], **k)
            out = out.to(torch.bfloat16) if isinstance(out, torch.Tensor) else out
        else:
            out = orig(*a, **k)
        log(f"{name} {time.time()-t:.1f}s")
        if hasattr(out, "__next__"):
            def gen():
                for i, x in enumerate(out):
                    save(f"{name}_out{i}", cpu(x))
                    yield x
            return gen()
        save(f"{name}_out", cpu(out))
        return out
    return call
for attr in ("upsampler", "video_decoder", "audio_decoder"):
    blk = getattr(pipe, attr)
    try:
        setattr(pipe, attr, type("Wrapped", (), {"__call__": staticmethod(wrap_block(blk, attr)),
                                                 "__getattr__": lambda self, n, b=blk: getattr(b, n)})())
    except Exception as e:
        log("wrap failed", attr, e)

meta = dict(image=args.image, strength=args.strength, prompt=PROMPT, seed=args.seed, height=args.height, width=args.width, frames=args.frames, fps=args.fps,
            vae=args.vae, torch=torch.__version__, threads=args.threads)
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
log("generate", meta)
with torch.inference_mode():
    from ltx_pipelines.utils.types import ImageConditioningInput
    imgs = [ImageConditioningInput.at(args.image, 0, args.strength)] if args.image else []
    chunks, nf, tc = pipe.stream_chunks(prompt=PROMPT, seed=args.seed, height=args.height, width=args.width,
                                        num_frames=args.frames, frame_rate=args.fps, images=imgs, vae_dtype=torch.float32)
    from ltx_pipelines.utils.media_io import encode_media_segments
    encode_media_segments(chunks, fps=args.fps, output_path=os.path.join(args.out, "out.mp4"),
                          video_chunks_number=None, color_space=None, video_marker=None, audio_marker=None)
log("DONE total", f"{time.time()-T0:.1f}s")
