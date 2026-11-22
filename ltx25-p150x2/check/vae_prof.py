"""ltx1: per-block timing of the LTX-2.5 conv VAE decoder on 1x2 (synchronized), at a given shape."""
import os, time, json
import torch, ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
from models.tt_dit.models.vae.vae_ltx import LTXVideoVAEAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
AUX = os.environ["LTX_AUX_CHECKPOINT"]
F, H, W = int(os.environ.get("F", 121)), int(os.environ.get("H", 704)), int(os.environ.get("W", 1280))
params = get_updated_device_params({"trace_region_size": 0, "l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D})
fc = params.pop("fabric_config"); params.pop("fabric_tensix_config", None)
ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **params)
ccl = CCLManager(mesh, num_links=int(os.environ.get("LINKS", 2)), topology=ttnn.Topology.Linear)
dit_pc = DiTParallelConfig(cfg_parallel=ParallelFactor(factor=1, mesh_axis=0), sequence_parallel=ParallelFactor(factor=1, mesh_axis=0), tensor_parallel=ParallelFactor(factor=2, mesh_axis=1))
hp = int(os.environ.get("HP", 2))
vae_pc = VaeHWParallelConfig(height_parallel=ParallelFactor(factor=hp, mesh_axis=1), width_parallel=ParallelFactor(factor=2 // hp, mesh_axis=0 if hp == 2 else 1))
vae = LTXVideoVAEAdapter(AUX, mesh_device=mesh, vae_parallel_config=vae_pc, vae_ccl_manager=ccl, dit_parallel_config=dit_pc, num_frames=F, height=H, width=W)
vae.reload_decoder()
dec = vae.decoder
lat = torch.randn(1, 128, (F - 1) // 8 + 1, H // 32, W // 32, generator=torch.Generator().manual_seed(0))
times = {}
def wrap(name, mod):
    orig = mod.forward
    def f(*a, **k):
        ttnn.synchronize_device(mesh); t = time.time(); r = orig(*a, **k); ttnn.synchronize_device(mesh)
        times[name] = times.get(name, 0) + time.time() - t; return r
    mod.forward = f
wrap("conv_in", dec.conv_in)
for i, b in enumerate(dec.up_blocks):
    wrap(f"up{i}_{type(b).__name__}", b)
wrap("norm_out", dec.norm_out); wrap("conv_out", dec.conv_out)
for it in range(2):
    times.clear(); ttnn.synchronize_device(mesh); t = time.time(); out = dec(lat, output_type="float"); tot = time.time() - t
    logger.info(f"VAEPROF iter {it} total {tot:.2f}s blocks {json.dumps({k: round(v, 2) for k, v in times.items()})}")
torch.save(out.float()[:, :, ::8].clone(), os.environ.get("SAVE", "/work/vaeprof_out.pt")); ttnn.close_mesh_device(mesh); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("VAEPROF_DONE")
