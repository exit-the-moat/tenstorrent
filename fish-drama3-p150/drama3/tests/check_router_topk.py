# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: the one-core router top-k kernel (tt/moe_fused_decode.py) vs ttnn.topk: random scores (with
forced ties) and every MoE layer's real router scores of a decode step; then the traced device time of the MoE
block with and without it.

    (fast env) python -m models.demos.blackhole.drama3.tests.check_router_topk [--out f.json]
"""
import argparse
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port5/check_router_topk.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.moe_fused_decode import RouterTopK
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
        rk = RouterTopK(mesh)

        def both(s_dev):
            v0, i0 = ttnn.topk(s_dev, k=8, dim=-1)
            v1, i1 = rk(s_dev)
            v0, v1 = ttnn.to_torch(v0).float().flatten()[:8], ttnn.to_torch(v1).float().flatten()[:8]
            i0 = ttnn.to_torch(ttnn.to_layout(i0, ttnn.ROW_MAJOR_LAYOUT)).flatten()[:8].long()
            i1 = ttnn.to_torch(i1).flatten()[:8].long()
            return v0, i0, v1, i1

        rnd = dict(n=0, values_equal=0, ids_equal=0, id_sets_equal=0)
        for seed in range(50):
            g = torch.Generator().manual_seed(seed)
            s = torch.randn(1, 1, 1, 256, generator=g) * (3 if seed % 2 else 0.05)  # small scale: many bf16 ties
            v0, i0, v1, i1 = both(ttnn.from_torch(s.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh))
            rnd["n"] += 1
            rnd["values_equal"] += int(torch.equal(v0, v1))
            rnd["ids_equal"] += int(torch.equal(i0, i1))
            rnd["id_sets_equal"] += int(set(i0.tolist()) == set(i1.tolist()))
        res["random"] = rnd
        print("random", json.dumps(rnd), flush=True)

        cfg = json.load(open(f"{MODEL}/config.json"))
        wmap = weight_map(MODEL)
        keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
        t = read_tensors(MODEL, wmap, keys)
        E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load("/opt/ep0/day3/close.rows.pt"))
        cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        bb.reset()
        T = E.shape[0] - 1
        _, _, per_layer = bb.prefill(E[:T], collect=True)  # host [T, H] per layer
        per_layer = [p_[T - 1] for p_ in per_layer]
        real = dict(n=0, values_equal=0, ids_equal=0, id_sets_equal=0)
        x = ttnn.from_torch(E[T - 1].reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        for i, layer in enumerate(bb.layers):
            h_in = x if i == 0 else ttnn.from_torch(per_layer[i - 1].reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            xn = layer.ffn_norm(h_in, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            r = layer.feed_forward.router
            s = ttnn.linear(ttnn.reshape(xn, (1, 1, 1, args.dim)), r.proj_weight, compute_kernel_config=r.compute_kernel_config)
            v0, i0, v1, i1 = both(s)
            real["n"] += 1
            real["values_equal"] += int(torch.equal(v0, v1))
            real["ids_equal"] += int(torch.equal(i0, i1))
            real["id_sets_equal"] += int(set(i0.tolist()) == set(i1.tolist()))
        res["real_layers"] = real
        print("real", json.dumps(real), flush=True)

        # traced MoE block, original vs fused top-k
        bb._stage(E[-1], E.shape[0])
        layer = bb.layers[0]

        def moe():
            xn = layer.ffn_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            return layer.feed_forward.forward(xn, mode="decode")

        from models.demos.blackhole.drama3.tt import moe_fused_decode

        for name in ("orig", "fused"):
            if name == "fused":
                os.environ["DRAMA3_MOE_FUSED_TOPK"] = "1"
                moe_fused_decode.install_topk([layer], mesh)
            out = ttnn.to_torch(moe()).float().flatten()
            res[f"{name}_out_head"] = out[:4].tolist()
            res.setdefault("outs", {})[name] = out
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            moe()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            res[f"{name}_moe_ms"] = round((time.time() - t0) / 100 * 1e3, 4)
            ttnn.release_trace(mesh, tid)
            print(name, "moe block traced", res[f"{name}_moe_ms"], "ms", flush=True)
        o = res.pop("outs")
        res["moe_out_equal"] = bool(torch.equal(o["orig"], o["fused"]))
        res["moe_out_maxabs"] = float((o["orig"] - o["fused"]).abs().max())
        print("moe out equal", res["moe_out_equal"], res["moe_out_maxabs"], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
