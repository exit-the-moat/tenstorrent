import os, time, torch, ttnn
for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3").items(): os.environ.setdefault(k, v)
from models.demos.blackhole.drama3.tests.profile_moe_decode import timed
from models.demos.blackhole.drama3.tt.device_check import build_layer
from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=64 << 20)
args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
layer = build_layer(mesh, args, 0, args.weight_cache_path(), weight_map(args.CKPT_DIR))[0]
moe = layer.feed_forward; W = moe.experts.weights
x = ttnn.from_torch(torch.randn(1, 1, 1, 2048) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
r = moe.router(x); sp = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT)
di4 = ttnn.from_torch(torch.randn(1, 256, 1, 512) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
def gu(b): return lambda: ttnn.sparse_matmul(x, W.gate_up_proj, sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(1, 1024, in0_block_w=b), dtype=ttnn.bfloat16)
def dn(b): return lambda: ttnn.sparse_matmul(di4, W.down_proj, sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(1, 2048, in0_block_w=b), is_input_a_sparse=True, dtype=ttnn.bfloat16)
ref_g = ttnn.to_torch(gu(1)()).float(); ref_d = ttnn.to_torch(dn(1)()).float()
for b in (1, 2, 4, 8, 16, 32):
    try:
        o, ms = timed(mesh, gu(b)); d = (ttnn.to_torch(o).float() - ref_g).abs().max().item(); print("gate_up in0_block_w", b, round(ms, 3), "ms maxdiff", d, flush=True)
    except Exception as e: print("gate_up", b, "ERR", str(e).splitlines()[0][:120], flush=True)
for b in (1, 2, 4, 8, 16):
    try:
        o, ms = timed(mesh, dn(b)); d = (ttnn.to_torch(o).float() - ref_d).abs().max().item(); print("down in0_block_w", b, round(ms, 3), "ms maxdiff", d, flush=True)
    except Exception as e: print("down", b, "ERR", str(e).splitlines()[0][:120], flush=True)
ttnn.close_device(mesh)
