"""Episode 0 Day 0 smoke test: open the p150 with the freshly built ttnn and check one matmul against torch."""
import time

import torch
import ttnn

t0 = time.time()
device = ttnn.open_device(device_id=0)
print("device opened in %.1fs, arch=%s, grid=%s" % (time.time() - t0, device.arch(), device.compute_with_storage_grid_size()), flush=True)

torch.manual_seed(0)
a = torch.randn(1024, 2048, dtype=torch.bfloat16)
b = torch.randn(2048, 512, dtype=torch.bfloat16)
ref = (a.float() @ b.float())

ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(ttnn.matmul(ta, tb)).float()

pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print("matmul 1024x2048x512 PCC=%.5f" % pcc, flush=True)
ttnn.close_device(device)
print("SMOKE_OK" if pcc > 0.99 else "SMOKE_BAD", flush=True)
