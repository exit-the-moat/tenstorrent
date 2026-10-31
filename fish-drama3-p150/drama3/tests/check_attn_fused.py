# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: the fused one-token attention pre-SDPA step (tt/attn_fused_decode.py) vs the op chain on every
backbone attention layer after a voice-prompt prefill (same input row, same position), then the traced device time of
one attention mixer (norm + attention) for both.

    (fast env) python -m models.demos.blackhole.drama3.tests.check_attn_fused [--out f.json]
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


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port5/check_attn_fused.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt import attn_fused_decode as A
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
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
        _, _, per_layer = bb.prefill(E[:T], collect=True)
        bb._stage(E[T], T)
        att_layers = [i for i, l in enumerate(bb.layers) if l.is_full_attention]
        up = lambda r: ttnn.from_torch(r.reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731

        def run(i):
            layer = bb.layers[i]
            xn = layer.attention_norm(up(per_layer[i - 1][T - 1]), mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            out = layer.attention.forward(xn, bb.cos_buf, bb.sin_buf, position_tensor=bb.pos_buf, page_table=bb.page_table)
            return ttnn.to_torch(out).float().flatten()[: args.dim]

        base = {i: run(i) for i in att_layers}
        layer = bb.layers[att_layers[0]]

        def mixer():
            xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            return layer.attention.forward(xn, bb.cos_buf, bb.sin_buf, position_tensor=bb.pos_buf, page_table=bb.page_table)

        def traced_ms():
            mixer()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            mixer()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ttnn.release_trace(mesh, tid)
            return round((time.time() - t0) / 100 * 1e3, 4)

        res["chain_attn_ms"] = traced_ms()
        os.environ["DRAMA3_ATTN_FUSED_PRE"] = "1"
        A.install([bb.layers[i].attention for i in att_layers], mesh)
        fused = {i: run(i) for i in att_layers}
        res["layers"] = {i: dict(pcc=round(pcc(fused[i], base[i]), 6), maxabs=round(float((fused[i] - base[i]).abs().max()), 4),
                                 bit_equal=bool(torch.equal(fused[i], base[i]))) for i in att_layers}
        for i in att_layers:
            print(i, json.dumps(res["layers"][i]), flush=True)
        res["fused_attn_ms"] = traced_ms()
        print("attention mixer traced: chain", res["chain_attn_ms"], "fused", res["fused_attn_ms"], "ms", flush=True)
        if os.environ.get("CHECK_POST", "0") == "1":  # + the post-SDPA kernel (re-installed over the original forwards)
            os.environ["DRAMA3_ATTN_FUSED_POST"] = "1"
            A.install([bb.layers[i].attention for i in att_layers], mesh)
            post = {i: run(i) for i in att_layers}
            res["layers_post"] = {i: dict(pcc=round(pcc(post[i], base[i]), 6), maxabs=round(float((post[i] - base[i]).abs().max()), 4)) for i in att_layers}
            for i in att_layers:
                print("post", i, json.dumps(res["layers_post"][i]), flush=True)
            res["fused_post_attn_ms"] = traced_ms()
            print("attention mixer traced: fused pre + post", res["fused_post_attn_ms"], "ms", flush=True)
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
