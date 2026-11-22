# ltx1: does ttnn SDPA accept a key-padding mask broadcast over queries ((1,1,32,Sk) instead of (1,1,Sq,Sk))?
import torch, ttnn
dev = ttnn.open_device(device_id=0)
B, H, Sq, Sk, D, nvalid = 1, 4, 512, 512, 128, 500
q, k, v = (torch.randn(B, H, n, D) for n in (Sq, Sk, Sk))
full = torch.zeros(1, 1, Sq, Sk); full[..., nvalid:] = float("-inf")
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=full)
tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False)
def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten(); a, b = a - a.mean(), b - b.mean(); return float(a @ b / (a.norm() * b.norm()))
for name, m in (("full", full), ("rows32", full[:, :, :32]), ("rows1", full[:, :, :1])):
    try:
        o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(tt(q), tt(k), tt(v), is_causal=False, attn_mask=tt(m), program_config=cfg)).float()
        print(name, tuple(m.shape), "PCC", round(pcc(o, ref), 5), flush=True)
    except Exception as e:
        print(name, tuple(m.shape), "ERROR", str(e).splitlines()[0][:200], flush=True)
ttnn.close_device(dev)
print("MASKTEST_DONE")
