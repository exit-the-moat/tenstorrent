# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track B: the batch MoE decode (tt/moe_fused_decode.py forward_batch: RouterTopKB + batched gather
sparse_matmuls + MoECombineB) vs the ttnn batched path (DRAMA3_MOE_FUSED_BATCH=0) and vs the one-token fused path
row by row, on one layer with B real normed rows; then the traced time of the MoE block for both batch paths.

    (interactive env) python -m models.demos.blackhole.drama3.tests.check_moe_batch [--layer 0] [--sizes 2,4,8]
"""
import argparse
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"
MODEL = "/models/drama-3"


def h(t):
    return ttnn.to_torch(t).float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sizes", default="2,4,8")
    ap.add_argument("--out", default="/opt/ep0/port6/check_moe_batch.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    sizes = [int(x) for x in a.sizes.split(",")]
    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load("/opt/ep0/day3/close.rows.pt"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = dict(layer=a.layer)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        layer = bb.layers[a.layer]
        moe = layer.feed_forward
        Bmax = max(sizes)
        rows = []
        for b in range(Bmax):  # real normed rows: frame embeddings through this layer's ffn norm
            bb._stage(E[-1 - b], 700 + b)
            rows.append(h(layer.ffn_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})).reshape(-1)[:2048])
        solo = []
        for b in range(Bmax):
            x1 = ttnn.from_torch(rows[b].reshape(1, 1, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            solo.append(h(moe.forward(x1, mode="decode")).reshape(-1)[:2048])
        for B in sizes:
            xb = ttnn.from_torch(torch.stack(rows[:B]).reshape(1, B, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            outs, ms = {}, {}
            for name, flag, dd in (("ttnn", "0", "0"), ("fused", "1", "0"), ("dedup", "1", "1")):  # port10: + dedup
                os.environ["DRAMA3_MOE_FUSED_BATCH"] = flag
                os.environ["DRAMA3_MOE_BATCH_DEDUP"] = dd
                outs[name] = h(moe.forward(xb, mode="decode")).reshape(B, -1)[:, :2048]
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                o = moe.forward(xb, mode="decode")
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(50):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms[name] = round((time.time() - t0) / 50 * 1e3, 4)
                ttnn.release_trace(mesh, tid)
            os.environ["DRAMA3_MOE_FUSED_BATCH"] = "0"
            os.environ["DRAMA3_MOE_BATCH_DEDUP"] = "0"
            r = dict(ms=ms)
            for name in outs:
                r[name] = dict(
                    pcc_vs_solo_min=round(min(pcc(outs[name][b], solo[b]) for b in range(B)), 6),
                    maxdiff_vs_solo=round(max(float((outs[name][b] - solo[b]).abs().max()) for b in range(B)), 5),
                    finite=bool(torch.isfinite(outs[name]).all()),
                )
            r["fused_vs_ttnn_pcc_min"] = round(min(pcc(outs["fused"][b], outs["ttnn"][b]) for b in range(B)), 6)
            r["dedup_vs_fused_pcc_min"] = round(min(pcc(outs["dedup"][b], outs["fused"][b]) for b in range(B)), 6)
            r["dedup_vs_fused_maxdiff"] = round(float((outs["dedup"] - outs["fused"]).abs().max()), 5)
            res[B] = r
            print(B, json.dumps(r), flush=True)
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
