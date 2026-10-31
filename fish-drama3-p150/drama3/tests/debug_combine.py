# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 debug: MoECombine on one layer's real tensors vs torch from the same device inputs."""
import json
import os

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def main():
    from models.demos.blackhole.drama3.tt import moe_fused_decode as M
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        cfg = json.load(open(f"{MODEL}/config.json"))
        cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), weight_map(MODEL), cand)
        m = bb.layers[0].feed_forward
        torch.manual_seed(0)
        x = ttnn.from_torch((torch.randn(1, 1, args.dim) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        x = bb.layers[0].ffn_norm(x, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
        H = args.dim
        x4, x3 = ttnn.reshape(x, (1, 1, 1, H)), ttnn.reshape(x, (1, 1, H))
        vals, idx = m.router.topk(x4)
        down = M._down_indexed(x4, idx, m.experts.weights, m.config, m.experts.decode_sparsity)
        g = ttnn.linear(x3, m.shared.gate_weight)
        sh = m.shared.mlp.forward(x3)
        for n_, t in (("vals", vals), ("idx", idx), ("down", down), ("g", g), ("sh", sh)):
            print(n_, tuple(t.shape), t.dtype, t.layout, t.memory_config().buffer_type, t.memory_config().memory_layout, flush=True)
        D, V, G, S = (ttnn.to_torch(t).float() for t in (down, vals, g, sh))
        print("down rows finite", torch.isfinite(D).all().item(), "row0 absmax", D[..., 0, :].abs().max().item(), flush=True)
        ref = (V.flatten()[:8, None] * D.reshape(8, -1, H)[:, 0, :]).sum(0) + torch.sigmoid(G.flatten()[0]) * S.flatten()[:H]
        out = ttnn.to_torch(M.MoECombine(mesh)(down, vals, sh, g)).float().flatten()[:H]
        print("out finite", torch.isfinite(out).all().item(), "nan count", int(torch.isnan(out).sum()), flush=True)
        print("ref[:6]", ref[:6].tolist(), flush=True)
        print("out[:6]", out[:6].tolist(), flush=True)
        f = torch.isfinite(out)
        print("maxabs on finite", (out[f] - ref[f]).abs().max().item() if f.any() else None, "first bad", torch.nonzero(~f).flatten()[:10].tolist(), flush=True)
        orig = ttnn.to_torch(m.forward(x, mode="decode")).float().flatten()[:H]
        print("ref vs orig maxabs", (ref - orig).abs().max().item(), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
