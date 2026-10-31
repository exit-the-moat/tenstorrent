import torch, ttnn
t = ttnn.from_torch(torch.randn(64,64).bfloat16(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
print("OK", ttnn.to_torch(t).shape)
