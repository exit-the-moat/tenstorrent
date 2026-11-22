# Build a small "monolith-style" aux checkpoint for tt_dit's LTX loaders: vae.* (conv or diff VAE), audio_vae.*, vocoder.*
# plus a merged metadata config, so the VAE/audio loaders never load_file() the 42 GB transformer file.
import json, sys
from safetensors import safe_open
from safetensors.torch import save_file
R = "/models/ltx25"
kind = sys.argv[1]  # conv | diff
out = sys.argv[2]
vae_f = f"{R}/vae/ltx-2.5-video-vae-conv-bf16.safetensors" if kind == "conv" else f"{R}/vae/ltx-2.5-video-vae-bf16.safetensors"
sd, cfg = {}, {}
with safe_open(f"{R}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "pt") as f:
    m = f.metadata(); cfg.update(json.loads(m["config"])); mv = m["model_version"]
with safe_open(vae_f, "pt") as f:
    cfg.update(json.loads(f.metadata()["config"]))
    for k in f.keys(): sd["vae." + k] = f.get_tensor(k)
with safe_open(f"{R}/vae/ltx-2.5-audio-vae-bf16.safetensors", "pt") as f:
    cfg.update(json.loads(f.metadata()["config"]))
    for k in f.keys(): sd[k] = f.get_tensor(k)
save_file(sd, out, metadata={"config": json.dumps(cfg), "model_version": mv, "note": f"ltx1 aux ({kind} VAE + audio VAE + vocoder) from Lightricks/LTX-2.5 rev 49b093a"})
print("DONE", out, len(sd), sorted(cfg.keys()))
