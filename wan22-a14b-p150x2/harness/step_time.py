"""Episode 4: where does a denoising step go? Wraps combined_step (both CFG forwards + lerp) and the UniPC solver
step with synchronized timers, runs N steps at 480p/81 frames with the EP4_* config, prints per-step pieces.
  python step_time.py --links 4 --steps 6 [--traced 0|1]
"""
import argparse
import json
import time

import ttnn
from tests.scripts.common import get_updated_device_params

from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--links", type=int, default=4)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--traced", type=int, default=0)
ap.add_argument("--sync", type=int, default=1, help="1 = synchronize around each piece (changes overlap)")
a = ap.parse_args()

params = get_updated_device_params({"trace_region_size": 200_000_000 if a.traced else 0,
                                    "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config")
params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED,
                       ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
cfg = WanPipelineConfig.default(
    mesh_shape=mesh.shape, dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(2, 1)),
    vae_parallel_config=VaeHWParallelConfig.from_tuples(height=(2, 1), width=(1, 0)),
    encoder_parallel_config=EncoderParallelConfig.from_tuple((2, 1)),
    num_links=a.links, dynamic_load=False, topology=ttnn.Topology.Linear, is_fsdp=False,
    checkpoint_name="/models/wan22-a14b", height=480, width=832, num_frames=81,
)
pipe = WanPipeline(device=mesh, config=cfg, run_warmup=False)
pipe(prompts=["warmup"], num_inference_steps=2, guidance_scale=2, guidance_scale_2=2, traced=bool(a.traced))

T = {"fwd": [], "solver": [], "step_wall": []}
orig_cs = WanTransformer3DModel.combined_step
orig_solver = pipe._solver.step
orig_step = pipe._step


def timed(name, fn):
    def w(*args, **kw):
        if a.sync:
            ttnn.synchronize_device(mesh)
        t0 = time.time()
        r = fn(*args, **kw)
        if a.sync:
            ttnn.synchronize_device(mesh)
        T[name].append(round(time.time() - t0, 3))
        return r
    return w


pipe._solver.step = timed("solver", orig_solver)
for st in pipe.transformer_states:
    st.model.combined_step = timed("fwd", st.model.combined_step)
pipe._step = timed("step_wall", orig_step)
t0 = time.time()
pipe(prompts=["A golden retriever puppy runs through a field of tall grass."], num_inference_steps=a.steps, seed=1,
     output_type="latent", traced=bool(a.traced))
ttnn.synchronize_device(mesh)
T["total"] = round(time.time() - t0, 2)
T["config"] = dict(links=a.links, steps=a.steps, traced=a.traced, sync=a.sync)
print("STEP_TIME " + json.dumps(T), flush=True)
with open("/out/step_time.jsonl", "a") as f:
    f.write(json.dumps(T) + "\n")
pipe.release_traces()
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
