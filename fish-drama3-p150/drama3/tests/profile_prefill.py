# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: where does the backbone prefill time go? Per layer and per part (token mixer vs MoE) on a warm run,
with a device sync after each piece.

    python -m models.demos.blackhole.drama3.tests.profile_prefill [--rows /opt/ep0/day3/close.rows.pt]
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
    ap.add_argument("--rows", default="/opt/ep0/day3/close.rows.pt")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import BLOCK, PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load(a.rows))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        bb.reset()
        bb.prefill(E)  # warm
        T = E.shape[0]
        bucket = bb.bucket_for(T)
        x = torch.zeros(1, bucket, E.shape[-1])
        x[0, :T] = E
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos, sin = bb.rope.get_prefill_rot_mats(0, bucket)
        chunk_pt = ttnn.from_torch(bb.page_table_host[:, : (T + BLOCK - 1) // BLOCK].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        csi = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        bb.reset()
        ttnn.synchronize_device(mesh)
        rows = []
        for i, layer in enumerate(bb.layers):
            t0 = time.time()
            xn = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=None)
            if layer.is_full_attention:
                att = layer.attention.forward(xn, cos, sin, page_table=bb.page_table, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi)
            else:
                att = layer.attention.forward(xn, mode="chunk", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=T)
            h = ttnn.add(x, att)
            ttnn.synchronize_device(mesh)
            t1 = time.time()
            ff = layer.ffn_norm(h, mode=Mode.PREFILL, norm_config=None)
            mo = layer.feed_forward.forward(ff, mode="prefill")
            ttnn.synchronize_device(mesh)
            t2 = time.time()
            # split the MoE: router + routed experts vs shared expert
            moe = layer.feed_forward
            ff4 = ttnn.reshape(ff, (1, 1, ff.shape[-2], ff.shape[-1]))
            r = moe.router(ff4)
            ex = moe.experts(ff4, r, mode="prefill")
            ttnn.synchronize_device(mesh)
            t3 = time.time()
            sh = moe.shared.forward(ff)
            ttnn.synchronize_device(mesh)
            t4 = time.time()
            y = ttnn.add(h, ttnn.reshape(mo, h.shape))
            for tt_ in (xn, att, ff, mo, r, ex, sh, h):
                ttnn.deallocate(tt_)
            ttnn.deallocate(x)
            x = y
            rows.append(dict(layer=i, type=layer.attention.__class__.__name__, mixer_s=round(t1 - t0, 3), moe_s=round(t2 - t1, 3), routed_s=round(t3 - t2, 3), shared_s=round(t4 - t3, 3)))
            print(json.dumps(rows[-1]), flush=True)
        tot = {k: round(sum(r[k] for r in rows), 2) for k in ("mixer_s", "moe_s", "routed_s", "shared_s")}
        print("TOTAL", json.dumps(tot), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
