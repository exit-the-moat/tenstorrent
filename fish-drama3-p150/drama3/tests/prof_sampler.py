# bb12: stage stamps of one FusedDraw (DRAMA3_SAMPLER_FUSED_PROFILE=1)
import torch, ttnn
from models.demos.blackhole.drama3.tt import device_sampler as ds
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
f = ds.FusedDraw(mesh, 1024, 30, 0.9)
g = torch.Generator().manual_seed(0)
for it in range(6):
    x = ttnn.from_torch(torch.randn(1, 1, 1, 1024, generator=g) * 3, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    n = ttnn.from_torch(torch.empty(1, 1, 1, 32).exponential_(1, generator=g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    c, _ = f(x, n)
    v = ttnn.to_torch(c).float().reshape(32, 32)
    raw = v.view(torch.int32)
    print("A", raw[0, 1:9].tolist(), "B", raw[0, 16:22].tolist(), "code", float(v[0, 0]), flush=True)
ttnn.close_mesh_device(mesh)
print("DONE")
