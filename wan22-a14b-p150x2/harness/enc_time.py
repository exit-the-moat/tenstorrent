"""Episode 4: honest umT5 timing on the cards (TP 2, as shipped in the pipeline), synchronized.

Cold = first encode_cfg in the process after the weights are on the card (includes program compile);
warm = later prompts. Each encode_cfg = prompt + negative prompt (CFG), 512 tokens. Also times the weight load from
the TT cache. Appends to /out/enc_time.jsonl.
"""
import json
import time

import ttnn
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.text_encoder import TextEncoder

params = get_updated_device_params({"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config")
params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
ccl = CCLManager(mesh_device=mesh, num_links=2, topology=ttnn.Topology.Linear)
enc = TextEncoder(checkpoint_name="/models/wan22-a14b", device=mesh, ccl_manager=ccl,
                  encoder_parallel_config=EncoderParallelConfig.from_tuple((2, 1)),
                  dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(2, 1)))
t0 = time.time()
enc.prepare()
ttnn.synchronize_device(mesh)
load_s = time.time() - t0
prompts = json.load(open("/ep4/harness/prompts8.json"))
neg = WanPipeline.DEFAULT_NEGATIVE_PROMPT
times = []
for j in prompts:
    t0 = time.time()
    p, n = enc.encode_cfg([j["prompt"]], [neg], cfg_enabled=True, num_videos_per_prompt=1)
    ttnn.synchronize_device(mesh)
    times.append(round(time.time() - t0, 3))
rec = dict(t=time.strftime("%F %T"), what="umT5 on cards TP2, prompt+negative, 512 tokens", load_from_cache_s=round(load_s, 2),
           cold_s=times[0], warm_s=times[1:], warm_median_s=sorted(times[1:])[len(times[1:]) // 2])
print(json.dumps(rec), flush=True)
with open("/out/enc_time.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("ENC_DONE")
