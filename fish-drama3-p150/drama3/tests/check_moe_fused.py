# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: the one-token MoE block of every layer with the fused pieces (router top-k, combine) vs the op
chain on real decode inputs (each layer's MoE input row from a voice-prompt prefill), plus the traced device time
of one layer's block for each variant.

    (fast env) python -m models.demos.blackhole.drama3.tests.check_moe_fused [--out f.json]
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
    ap.add_argument("--out", default="/opt/ep0/port5/check_moe_fused.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt import moe_fused_decode as M
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
        T = E.shape[0]
        _, _, per_layer = bb.prefill(E, collect=True)
        rows = [E[T - 1]] + [p_[T - 1] for p_ in per_layer[:-1]]  # each layer's input row (stands in for the MoE input)
        up = lambda r: ttnn.from_torch(r.reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731

        def run_all():
            outs = []
            for i, layer in enumerate(bb.layers):
                xn = layer.ffn_norm(up(rows[i]), mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                outs.append(ttnn.to_torch(layer.feed_forward.forward(xn, mode="decode")).float().flatten()[: args.dim])
            return outs

        layer0 = bb.layers[0]
        bb._stage(E[-1], T)

        def moe0():
            xn = layer0.ffn_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            return layer0.feed_forward.forward(xn, mode="decode")

        def traced_ms():
            moe0()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            moe0()
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

        base = run_all()
        res["orig_moe_ms"] = traced_ms()
        print("orig moe block", res["orig_moe_ms"], "ms", flush=True)
        variants = (("topk", "DRAMA3_MOE_FUSED_TOPK", M.install_topk), ("topk_combine", "DRAMA3_MOE_FUSED_COMBINE", M.install_combine),
                    ("topk_combine_swiglu", "DRAMA3_MOE_FUSED_SWIGLU", M.install_combine),
                    ("topk_combine_swiglu_emm", "DRAMA3_MOE_EXPERT_MM", M.install_combine))
        for name, env, inst in variants:
            os.environ[env] = "1"
            inst(bb.layers, mesh)
            outs = run_all()
            p = [pcc(o, b) for o, b in zip(outs, base)]
            mx = [float((o - b).abs().max()) for o, b in zip(outs, base)]
            res[name] = dict(pcc_min=round(min(p), 6), pcc_mean=round(sum(p) / len(p), 6), maxabs_max=round(max(mx), 4),
                             bit_equal_layers=sum(int(torch.equal(o, b)) for o, b in zip(outs, base)), moe_ms=traced_ms())
            print(name, json.dumps(res[name]), flush=True)
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
