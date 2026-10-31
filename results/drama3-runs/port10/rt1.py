import torch, ttnn
m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
print("open ok", flush=True)
for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
    x = ttnn.from_torch(torch.randn(1, 1, 2048, 2048), dtype=dt, layout=ttnn.TILE_LAYOUT, device=m)
    y = ttnn.to_torch(x)
    print("roundtrip ok", dt, flush=True)
z = ttnn.to_torch(ttnn.matmul(x, x))
print("matmul ok", flush=True)
ttnn.close_device(m)
