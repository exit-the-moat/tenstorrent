# ltx1: SDPA time at the 161-frame stage-2 shape: unmasked vs key-padding mask (one mask, chunked queries as in M3)
import time, torch, ttnn
dev = ttnn.open_device(device_id=0)
H, D, Sk, nvalid = 16, 128, 42880, 42848
cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), q_chunk_size=256, k_chunk_size=256, exp_approx_mode=False)
ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
q, k, v = tt(torch.randn(1, H, Sk, D)), tt(torch.randn(1, H, Sk, D)), tt(torch.randn(1, H, Sk, D))
def timeit(f, n=3):
    f(); ttnn.synchronize_device(dev); t = time.time()
    for _ in range(n): o = f(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev); return (time.time() - t) / n
t_un = timeit(lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, program_config=cfg, compute_kernel_config=ck))
for cq in (2048, 8192):
    m = torch.zeros(1, 1, cq, Sk); m[..., nvalid:] = float("-inf"); mt = tt(m)
    def masked():
        outs = []
        for i in range(0, Sk, cq):
            qi = q[:, :, i:min(i + cq, Sk), :]
            mm = mt if qi.shape[2] == cq else tt(torch.zeros(1, 1, qi.shape[2], Sk).masked_fill_(torch.arange(Sk) >= nvalid, float("-inf")))
            outs.append(ttnn.transformer.scaled_dot_product_attention(qi, k, v, attn_mask=mm, is_causal=False, program_config=cfg, compute_kernel_config=ck))
        o = ttnn.concat(outs, dim=2); [ttnn.deallocate(x) for x in outs]; return o
    print(f"masked chunk {cq}: {timeit(masked):.3f} s", flush=True)
print(f"unmasked: {t_un:.3f} s per SDPA call (H={H}, N={Sk})", flush=True)
ttnn.close_device(dev); print("BENCH_DONE")
