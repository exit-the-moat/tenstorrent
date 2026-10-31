"""Host <-> card bandwidth over the PCIe link: time ttnn.from_torch / to_torch of 1 GiB bf16 tensors (DRAM interleaved)."""
import time, json, torch, ttnn
dev = ttnn.open_device(device_id=0)
x = torch.randn(16, 1, 4096, 8192, dtype=torch.bfloat16)  # 1 GiB
gib = x.numel() * 2 / 2**30
res = {"bytes_gib": gib, "h2d_s": [], "d2h_s": []}
for i in range(4):
    t0 = time.time(); t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.synchronize_device(dev); res["h2d_s"].append(time.time() - t0)
    t0 = time.time(); y = ttnn.to_torch(t); res["d2h_s"].append(time.time() - t0)
    ttnn.deallocate(t)
ok = torch.equal(y, x)
res["roundtrip_equal"] = ok
best = lambda v: min(v[1:])
res["h2d_gib_s"] = gib / best(res["h2d_s"]); res["d2h_gib_s"] = gib / best(res["d2h_s"])
print("PCIE", json.dumps(res), flush=True)
ttnn.close_device(dev)
