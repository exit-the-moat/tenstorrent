import torch, ttnn
from models.demos.blackhole.drama3.host.codec import load_codec
c = load_codec()
al = [m.alpha.detach().flatten() for m in c.decoder.modules() if type(m).__name__ == "Snake1d"]
print("alpha min/max", min(float(a.min()) for a in al), max(float(a.max()) for a in al))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
for R in (3, 10, 30, 100, 300):
    x = torch.linspace(-R, R, 32 * 1024).reshape(1, 1, 32, 1024)
    for dt in (ttnn.float32, ttnn.bfloat16):
        xd = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
        y = ttnn.to_torch(ttnn.sin(xd)).float()
        xr = ttnn.to_torch(xd).float()
        print(R, dt, "max err vs torch.sin(rounded x)", float((y - torch.sin(xr)).abs().max()))
ttnn.close_device(mesh)
