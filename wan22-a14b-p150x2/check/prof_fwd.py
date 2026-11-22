"""Episode 4: one Wan DiT forward (cond only, high-noise expert) at 480p / 81 frames on the 1x2 mesh, for the tracy
device profiler. Config from the environment (EP4_*) and --links. Runs the forward R times; signposts fwd_start /
fwd_end around the last one. Run:  python -m tracy -r -v -o /ep4/prof/<tag> /ep4/check/prof_fwd.py --links 4
"""
import argparse
import time

import torch
import ttnn
from tests.scripts.common import get_updated_device_params

from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tensor import float32_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--links", type=int, default=4)
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--res", default="480p")
a = ap.parse_args()
W, H = (832, 480) if a.res == "480p" else (1280, 720)

params = get_updated_device_params({"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
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
    checkpoint_name="/models/wan22-a14b", height=H, width=W, num_frames=a.frames,
)
pipe = WanPipeline(device=mesh, config=cfg, run_warmup=False)
pipe._prepare_transformer(0)
model = pipe.transformer_states[0].model
torch.manual_seed(0)
F = (a.frames - 1) // 4 + 1
lat = torch.randn(1, 16, F, H // 8, W // 8)
prompt = tensor.from_torch(torch.randn(1, 1, 512, 4096) * 0.1, device=mesh, dtype=ttnn.bfloat16)
prompt = model.prepare_text_conditioning(prompt)
t = float32_tensor(torch.tensor([900.0]).reshape(1, 1, 1, 1), device=mesh)
spatial, N = model.preprocess_spatial_input_host(lat)
spatial_tt = tensor.from_torch(spatial, device=mesh, dtype=ttnn.bfloat16)
rc, rs, tm = model.get_rope_features(lat)
for i in range(a.repeat):
    last = i == a.repeat - 1
    if last:
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush the device profiler buffer so the measured forward fits
        from tracy import signpost

        signpost("fwd_start")
    t0 = time.time()
    o = model.inner_step(spatial_tt, prompt, rc, rs, tm, N, t)
    ttnn.synchronize_device(mesh)
    dt = time.time() - t0
    if last:
        signpost("fwd_end")
    print(f"forward {i}: {dt:.3f} s", flush=True)
    ttnn.deallocate(o)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("PROF_FWD_DONE")
