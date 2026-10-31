import os, torch, ttnn
for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3").items(): os.environ.setdefault(k, v)
from models.demos.blackhole.drama3.tests.profile_moe_decode import timed
from models.demos.blackhole.drama3.tt.device_check import build_layer
from models.demos.blackhole.drama3.tt.weight_mapping import weight_map, load_drama3_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config
def pcc(a,b):
    a,b=a.double().flatten(),b.double().flatten(); a,b=a-a.mean(),b-b.mean(); return float(a@b/(a.norm()*b.norm()))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=64 << 20)
args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
wm = weight_map(args.CKPT_DIR)
layer = build_layer(mesh, args, 0, args.weight_cache_path(), wm)[0]
moe = layer.feed_forward; W = moe.experts.weights
torch.manual_seed(0); xt = torch.randn(1, 1, 1, 2048) * 0.5
x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
r = moe.router(x); sp = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT)
act = torch.nonzero(ttnn.to_torch(r).float().reshape(-1)).reshape(-1).tolist()
sd = load_drama3_state_dict(args.CKPT_DIR, [0], wmap=wm)
g = torch.stack([sd[f"layers.0.mlp.experts.{e}.gate_proj.weight"].float() for e in act]); u = torch.stack([sd[f"layers.0.mlp.experts.{e}.up_proj.weight"].float() for e in act])
ref = torch.cat([u @ xt.reshape(-1).bfloat16().float(), g @ xt.reshape(-1).bfloat16().float()], -1)  # [8, 1024] up|gate, fp32
fp32 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False)
hifi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False)
for name, ckc in (("default", None), ("LoFi+fp32acc", fp32), ("HiFi2+fp32acc", hifi)):
    for b in (1, 4, 8, 16, 32):
        f = lambda: ttnn.sparse_matmul(x, W.gate_up_proj, sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(1, 1024, in0_block_w=b), compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        try:
            o, ms = timed(mesh, f)
            ot = ttnn.to_torch(o).float().reshape(256, -1)[act][:, :1024]
            print(f"{name:14s} block {b:2d} {ms:.3f} ms  PCC vs fp32 torch {pcc(ot, ref):.6f}", flush=True)
        except Exception as e: print(name, b, "ERR", str(e).splitlines()[0][:100], flush=True)
ttnn.close_device(mesh)
