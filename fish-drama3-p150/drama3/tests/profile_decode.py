# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: where do the ~65 ms of a traced backbone decode go? Traces one piece at a time (a GDN layer, an attention
layer, the MoE block, the token mixer alone, the final norm + head) and times 50 replays each.

    python -m models.demos.blackhole.drama3.tests.profile_decode
"""
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def main():
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load("/opt/ep0/day3/close.rows.pt"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        bb.reset()
        bb.prefill(E)
        T = E.shape[0]
        bb._stage(E[-1], T)
        gdn, att = bb.layers[0], bb.layers[3]

        def mixer(layer):
            def f():
                xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                if layer.is_full_attention:
                    return layer.attention.forward(xn, bb.cos_buf, bb.sin_buf, position_tensor=bb.pos_buf, page_table=bb.page_table)
                return layer.attention.forward(xn, mode="recurrent")
            return f

        def moe(layer):
            def f():
                xn = layer.ffn_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                return layer.feed_forward.forward(xn, mode="decode")
            return f

        def routed(layer):
            def f():
                x4 = ttnn.reshape(bb.x_buf, (1, 1, 1, args.dim))
                return layer.feed_forward.experts(x4, layer.feed_forward.router(x4), mode="decode")
            return f

        def shared(layer):
            return lambda: layer.feed_forward.shared.forward(bb.x_buf)

        def head():
            xn = bb.norm(bb.x_buf, mode=Mode.PREFILL)
            return ttnn.linear(xn, bb.head, dtype=ttnn.float32, compute_kernel_config=bb.head_ckc)

        pieces = {
            "gdn_layer_full": lambda: gdn.forward(bb.x_buf, mode="decode"),
            "attn_layer_full": lambda: att.forward(bb.x_buf, cos=bb.cos_buf, sin=bb.sin_buf, mode="decode", position_tensor=bb.pos_buf, page_table=bb.page_table),
            "gdn_mixer": mixer(gdn),
            "attn_mixer": mixer(att),
            "moe_block": moe(gdn),
            "moe_routed_only": routed(gdn),
            "moe_shared_only": shared(gdn),
            "final_norm_head": head,
        }
        res = {}
        for name, fn in pieces.items():
            out = fn()  # compile
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(50):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            res[name] = round((time.time() - t0) / 50 * 1e3, 3)
            ttnn.release_trace(mesh, tid)
            print(name, res[name], "ms", flush=True)
        n_gdn = sum(not l.is_full_attention for l in bb.layers)
        est = n_gdn * res["gdn_layer_full"] + (40 - n_gdn) * res["attn_layer_full"] + res["final_norm_head"]
        res["estimate_full_decode_ms"] = round(est, 1)
        print(json.dumps(res), flush=True)
        json.dump(res, open("/opt/ep0/day4/profile_decode.json", "w"), indent=1)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
