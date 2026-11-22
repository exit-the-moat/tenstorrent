# ltx1: can SDPA run unpadded at N=42848 (1339 tiles = 13 x 103) with chunk sizes that divide N? time + PCC vs torch on a slice
import time, torch, ttnn
dev = ttnn.open_device(device_id=0)
H, D, N = 16, 128, 42848
ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
torch.manual_seed(0)
qh, kh, vh = torch.randn(1, H, N, D), torch.randn(1, H, N, D), torch.randn(1, H, N, D)
q, k, v = tt(qh), tt(kh), tt(vh)
ref = torch.nn.functional.scaled_dot_product_attention(qh[:, :2, :512].float(), kh[:, :2].float(), vh[:, :2].float())
def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten(); a, b = a - a.mean(), b - b.mean(); return float(a @ b / (a.norm() * b.norm()))
for qc, kc in ((256, 256), (416, 416), (256, 416), (128, 416), (32 * 103, 416)):
    try:
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
        f = lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, program_config=cfg, compute_kernel_config=ck)
        o = f(); ttnn.synchronize_device(dev); t = time.time()
        for _ in range(3): ttnn.deallocate(o); o = f()
        ttnn.synchronize_device(dev); dt = (time.time() - t) / 3
        ot = ttnn.to_torch(o)[:, :2, :512].float()
        print(f"q{qc} k{kc}: {dt:.3f} s  PCC {pcc(ot, ref):.5f}", flush=True); ttnn.deallocate(o)
    except Exception as e:
        print(f"q{qc} k{kc}: ERROR {str(e).splitlines()[0][:160]}", flush=True)
ttnn.close_device(dev); print("BENCH_DONE")
