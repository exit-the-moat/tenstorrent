"""wan2: streaming host merge of a LoRA into one Wan expert, shard by shard (peak RAM ~ one shard).

For each safetensors shard of the fp32 expert: read every tensor in fp32, add scale*alpha/r * up @ down for LoRA-targeted
weights (own Wan-native -> diffusers key map, same as cpu_ref.py), cast to bf16, serialize, write, fsync, evict the
page cache and re-read to check the sha256. Writes the index + config next to it and merge_log.json.
  python merge_lora_stream.py --src /models/wan22-a14b/transformer --dst /work/merged/<name>/transformer --lora <file>
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
MAP = {"self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k", "self_attn.v": "attn1.to_v", "self_attn.o": "attn1.to_out.0",
       "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k", "cross_attn.v": "attn2.to_v", "cross_attn.o": "attn2.to_out.0",
       "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2"}
raw = load_file(a.lora)
deltas = {}
for k in raw:
    m = re.match(r"^(?:diffusion_model\.)?(blocks\.\d+)\.(.+?)\.lora_(down|A)(?:\.[^.]+)?\.weight$", k)
    if not m:
        continue
    up_k = k.replace(".lora_down.", ".lora_up.") if m.group(3) == "down" else k.replace(".lora_A.", ".lora_B.")
    down, up = raw[k].float(), raw[up_k].float()
    a_k = k.split(".lora_")[0] + ".alpha"
    alpha = float(raw[a_k].item()) if a_k in raw else float(down.shape[0])
    deltas[f"{m.group(1)}.{MAP[m.group(2)]}.weight"] = (down, up, a.scale * alpha / down.shape[0])
del raw
idx = json.load(open(os.path.join(a.src, "diffusion_pytorch_model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
log = dict(src=a.src, lora=a.lora, scale=a.scale, n_deltas=len(deltas), eff_scale=next(iter(deltas.values()))[2],
           shards={}, applied=0, t0=time.strftime("%F %T"))
applied = set()
total_size = 0
for sh in shards:
    t = time.time()
    out = {}
    with safe_open(os.path.join(a.src, sh), framework="pt") as f:
        for name in f.keys():
            w = f.get_tensor(name).float()
            if name in deltas:
                down, up, sc = deltas[name]
                w = w + sc * (up @ down)
                applied.add(name)
            out[name] = w.to(torch.bfloat16)
            del w
    blob = save(out, metadata={"format": "pt"})
    del out
    h = hashlib.sha256(blob).hexdigest()
    p = os.path.join(a.dst, sh)
    with open(p + ".tmp", "wb") as fo:
        fo.write(blob)
        fo.flush()
        os.fsync(fo.fileno())
    total_size += len(blob)
    del blob
    os.replace(p + ".tmp", p)
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    h2 = hashlib.sha256()
    with open(p, "rb") as fi:
        for c in iter(lambda: fi.read(1 << 24), b""):
            h2.update(c)
    ok = h2.hexdigest() == h
    log["shards"][sh] = dict(sha256=h, readback_ok=ok, s=round(time.time() - t, 1))
    print(f"{sh} {'OK' if ok else 'BAD'} {time.time() - t:.0f}s applied so far {len(applied)}", flush=True)
    if not ok:
        raise SystemExit("MERGE_FAILED readback mismatch " + sh)
missing = sorted(set(deltas) - applied)
log["applied"] = len(applied)
log["missing"] = missing[:10]
if missing:
    raise SystemExit(f"MERGE_FAILED {len(missing)} LoRA targets not found, e.g. {missing[:3]}")
idx2 = dict(idx)
idx2["metadata"] = {"total_size": total_size}
json.dump(idx2, open(os.path.join(a.dst, "diffusion_pytorch_model.safetensors.index.json"), "w"))
cfg = json.load(open(os.path.join(a.src, "config.json")))
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
log["t1"] = time.strftime("%F %T")
json.dump(log, open(os.path.join(a.dst, "merge_log.json"), "w"), indent=1)
print(f"MERGE_DONE applied {len(applied)}/{len(deltas)} eff_scale {log['eff_scale']}", flush=True)
