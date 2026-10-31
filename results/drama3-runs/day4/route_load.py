# expert load per layer for the 8 prompts, from the CPU side: replicate the router on host (fp32) with the golden-free path
import json, torch, yaml, os
os.environ.setdefault("HF_MODEL","/opt/ep0/drama3_shim"); os.environ.setdefault("TT_CACHE_PATH","/opt/ep0/cache/drama3")
import ttnn
from models.demos.blackhole.qwen36.tt.moe import prefill_grouped as G
from models.demos.blackhole.drama3.host.generate_fast import FastSession
from models.demos.blackhole.drama3.host.generate import reference_codes
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
Ms=[]
orig=G.grouped_prefill_forward
def wrap(h, r, w, c, m, valid_len=None):
    S,E=r.shape[-2], r.shape[-1]; rr=ttnn.to_torch(r).float().reshape(S,E)[:valid_len]
    Ms.append(int(torch.bincount(torch.topk(rr,8,-1).indices.reshape(-1),minlength=E).max()))
    return orig(h,r,w,c,m,valid_len)
import models.demos.blackhole.qwen36.tt.moe.experts as X
X.grouped_prefill_forward=wrap
lines=yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256<<20)
s=FastSession(mesh)
for ref in ("ref_long","ref_v2"):
    codes=reference_codes(f"/opt/ep0/voice/{ref}.wav","/opt/ep0/day3"); txt=open(f"/opt/ep0/voice/{ref}.txt").read().strip()
    for l in lines:
        Ms.clear(); rows=s.prompt(l["text"],[(txt,codes)]); s.bb.reset(); s.bb.prefill(s.embedder(rows))
        print(ref, l["id"], rows.shape[0], "max load per layer: max", max(Ms), "min", min(Ms), "sorted top5", sorted(Ms)[-5:], flush=True)
ttnn.close_device(mesh)
