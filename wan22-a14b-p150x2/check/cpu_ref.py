"""Episode 4: CPU fp32 reference of one Wan2.2 DiT forward (stock diffusers WanTransformer3DModel), block-streamed.

The fp32 checkpoint is 57 GB per expert; the host has 62 GB. The model is built on the meta device; the non-block
weights load up front, each transformer block loads its weights from the safetensors right before it runs and is
moved back to meta right after. Input file (torch.save dict, written by pcc_card.py):
  latent [1,16,F,H,W] fp32, t (float), prompt [1,512,4096], negative [1,512,4096] (optional), expert ("transformer" or
  "transformer_2"). Output: {"cond": [1,16,F,H,W], "uncond": ...} fp32.
"""
import argparse
import json
import os
import time

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import WanTransformer3DModel
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="/models/wan22-a14b")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--lora-mode", default="fp32", choices=["fp32", "none", "devfuse_bf16", "hostmerge_bf16"],
                help="wan2 diagnosis: fp32 = W + delta in fp32 (the reference); none = base only; devfuse_bf16 = simulate the on-card fuse (bf16 W, bf16 delta from bf16 factors, bf16 add); hostmerge_bf16 = bf16(W + delta) merged in fp32 then cast; all other weights bf16-rounded in the two bf16 modes")
a = ap.parse_args()
torch.set_num_threads(a.threads)

inp = torch.load(a.inputs)
sub = inp["expert"]
d = os.path.join(inp.get("ckpt", a.ckpt), sub)  # I2V inputs carry their checkpoint path
handles = {}
if inp.get("lightx2v_file"):
    # distill: weights from the lightx2v flat file (renamed to diffusers keys by tt_dit's loader, held in RAM as bf16),
    # model config from the base I2V checkpoint (same graph)
    from models.tt_dit.experimental.utils.lightx2v_loader import load_lightx2v_state_dict

    SD = load_lightx2v_state_dict("lightx2v/Wan2.2-Distill-Models", inp["lightx2v_file"], allow_download=False,
                                  local_dir=inp["lightx2v_dir"])
    idx = {k: None for k in SD}

    def get(name):
        return SD[name].float()
else:
    idx = json.load(open(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(d, f), framework="pt")
        return handles[f].get_tensor(name).float()


if inp.get("lora_file"):
    # wan2: base + LoRA (Wan2.2-Lightning, Wan-native keys), merged here in fp32: W' = W + scale * alpha/rank * up @ down.
    # Own key map (independent of tt_dit's adapter_loader, so a mapping bug there would show up as low PCC).
    from safetensors.torch import load_file

    _L = load_file(inp["lora_file"])
    _MAP = {"self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k", "self_attn.v": "attn1.to_v", "self_attn.o": "attn1.to_out.0",
            "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k", "cross_attn.v": "attn2.to_v", "cross_attn.o": "attn2.to_out.0",
            "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2"}
    _DELTA = {}
    for k in _L:
        if not k.endswith(".lora_down.weight"):
            continue
        base = k[: -len(".lora_down.weight")].replace("diffusion_model.", "")
        _, bi, rest = base.split(".", 2)
        down, up = _L[k].float(), _L[k.replace("lora_down", "lora_up")].float()
        alpha = float(_L[k.replace(".lora_down.weight", ".alpha")].item())
        _DELTA[f"blocks.{bi}.{_MAP[rest]}.weight"] = (down, up, float(inp.get("lora_scale", 1.0)) * alpha / down.shape[0])
    assert len(_DELTA) == 400, len(_DELTA)
    _base_get, _used = get, set()

    _bf = lambda x: x.to(torch.bfloat16).float()  # noqa: E731

    def get(name):
        w = _base_get(name)
        m = a.lora_mode
        if name in _DELTA:
            down, up, sc = _DELTA[name]
            _used.add(name)
            if m == "fp32":
                w = w + sc * (up @ down)
            elif m == "devfuse_bf16":
                w = _bf(_bf(w) + _bf(_bf(sc * up) @ _bf(down)))
            elif m == "hostmerge_bf16":
                w = _bf(w + sc * (up @ down))
        elif m in ("devfuse_bf16", "hostmerge_bf16") and name.endswith(".weight") and w.dim() >= 2:
            w = _bf(w)
        return w

    print(f"LoRA {inp['lora_file']}: {len(_DELTA)} deltas, scale*alpha/rank = {next(iter(_DELTA.values()))[2]}", flush=True)

cfg = WanTransformer3DModel.load_config(d)
with init_empty_weights():
    model = WanTransformer3DModel.from_config(cfg)
model.eval()

for name in idx:
    if not name.startswith("blocks."):
        set_module_tensor_to_device(model, name, "cpu", value=get(name))
# buffers (rope freqs) are created in __init__; recompute them on cpu if they were put on meta
for mn, mod in model.named_modules():
    for bn, buf in list(mod.named_buffers(recurse=False)):
        if buf.is_meta:
            raise SystemExit(f"meta buffer {mn}.{bn}: rebuild needed")


def pre_hook(i):
    def f(block, args):
        t = time.time()
        for pn, _ in block.named_parameters():
            set_module_tensor_to_device(block, pn, "cpu", value=get(f"blocks.{i}.{pn}"))
        f.load_s = time.time() - t
    return f


def post_hook(i):
    def f(block, args, out):
        block.to("meta")
        print(f"block {i} done {time.strftime('%H:%M:%S')}", flush=True)
    return f


for i, b in enumerate(model.blocks):
    b.register_forward_pre_hook(pre_hook(i))
    b.register_forward_hook(post_hook(i))

lat = inp["latent"].float()
if inp.get("cond") is not None:  # I2V: latent first, then mask + encoded image (diffusers WanImageToVideoPipeline order)
    lat = torch.cat([lat, inp["cond"].float()], dim=1)
embs = [inp["prompt"].float().reshape(1, -1, 4096)]
if inp.get("negative") is not None:
    embs.append(inp["negative"].float().reshape(1, -1, 4096))
hs = lat.expand(len(embs), *lat.shape[1:]).contiguous()
ehs = torch.cat(embs, 0)
t = torch.full((len(embs),), float(inp["t"]))
t0 = time.time()
with torch.no_grad():
    out = model(hidden_states=hs, timestep=t, encoder_hidden_states=ehs, return_dict=False)[0]
res = {"cond": out[0:1].clone()}
if len(embs) > 1:
    res["uncond"] = out[1:2].clone()
if inp.get("lora_file"):
    assert len(_used) == 400, f"only {len(_used)} of 400 LoRA deltas applied"
    print(f"lora-mode {a.lora_mode}", flush=True)
torch.save(res, a.out)
print(f"CPU_REF_DONE {a.out} {time.time()-t0:.0f}s", flush=True)
