# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track B: the batch GDN decode kernel (v4, tt/gdn_fused_decode.py forward_batch) vs the one-token v3 kernel
on ONE layer after a real prompt prefill: B different decode rows (the next B frames' embeddings, each from the
same start state), v3 once per row, v4 once for all rows on the same in-projection rows. Expect bit-identical
outputs, states and conv partial sums per row. Then the traced device time of v3 x B and of v4.

    (interactive env) python -m models.demos.blackhole.drama3.tests.check_gdn_batch [--layer 0] [--sizes 2,3,4,8]
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


def h(t):
    return ttnn.to_torch(t).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sizes", default="2,3,4,8")
    ap.add_argument("--out", default="/opt/ep0/port6/check_gdn_batch.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.gdn_fused_decode import FusedGDNDecode
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    sizes = [int(x) for x in a.sizes.split(",")]
    Bmax = max(sizes)
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
        bb.reset()
        T = E.shape[0] - Bmax
        bb.prefill(E[:T])
        layer = bb.layers[a.layer]
        dn, w = layer.attention, layer.attention.weights
        ckc = dn.compute_kernel_config_decode
        fused = getattr(dn, "fused_decode", None) or FusedGDNDecode(dn, mesh)
        Q = list(dn.split_conv_state)
        solo = (dn.recurrent_state, dn.split_conv_state)
        S0, Q0 = h(dn.recurrent_state), [h(q) for q in Q]  # [1, 32, 128, 128], 3 x [1, 1, 8192]
        megas, ref = [], []
        for b in range(Bmax):  # v3 per row from the same start state
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(S0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dn.recurrent_state)
            for q, q0 in zip(Q, Q0):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(q0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), q)
            bb._stage(E[T + b], T + b)
            xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
            mega = ttnn.linear(xn, w.mega_fused_weight, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc)
            megas.append(h(mega).reshape(-1)[:12352])
            fused(mega)
            ttnn.synchronize_device(mesh)
            ref.append(dict(o=h(fused.out).reshape(-1)[:4096], S=h(dn.recurrent_state)[0], Q=[h(q).reshape(-1)[:8192] for q in Q]))
        dev = lambda x, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc)  # noqa: E731
        for B in sizes:
            st = dev(S0.repeat(B, 1, 1, 1))
            qs = []
            for q0 in Q0:
                qq = torch.zeros(1, 2 * B, 8192)
                qq[0, 0::2] = q0.reshape(1, 8192)
                qs.append(dev(qq))
            mega_b = dev(torch.stack(megas[:B]).reshape(1, B, 12352), ttnn.L1_MEMORY_CONFIG)
            dn.recurrent_state, dn.split_conv_state = st, qs
            prog, io = fused._program_batch(mega_b, B)
            ttnn.generic_op(io, prog)
            ttnn.synchronize_device(mesh)
            o = h(fused.out_b[B]).reshape(B, 4096)
            S = h(st)
            Qb = [h(q).reshape(2 * B, 8192) for q in qs]
            r = {}
            for b in range(B):
                r[f"row{b}"] = dict(
                    o_identical=bool(torch.equal(o[b], ref[b]["o"])), o_maxdiff=float((o[b] - ref[b]["o"]).abs().max()),
                    S_identical=bool(torch.equal(S[b], ref[b]["S"])), S_maxdiff=float((S[b] - ref[b]["S"]).abs().max()),
                    Q_identical=all(torch.equal(Qb[k][2 * b], ref[b]["Q"][k]) for k in range(3)),
                    Q_odd_rows_zero=all(bool((Qb[k][2 * b + 1] == 0).all()) for k in range(3)),
                )
            # traced device time: v4 for B rows vs v3 once
            dn.recurrent_state, dn.split_conv_state = st, qs
            def run():
                p_, io_ = fused._program_batch(mega_b, B)
                ttnn.generic_op(io_, p_)
            run()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            run()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(50):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            r["v4_kernel_ms"] = round((time.time() - t0) / 50 * 1e3, 4)
            ttnn.release_trace(mesh, tid)
            res[B] = r
            print(B, json.dumps(r), flush=True)
            dn.recurrent_state, dn.split_conv_state = solo
        mega1 = dev(megas[0].reshape(1, 1, 12352), ttnn.L1_MEMORY_CONFIG)
        fused(mega1)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fused(mega1)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(50):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        res["v3_kernel_ms"] = round((time.time() - t0) / 50 * 1e3, 4)
        print("v3", res["v3_kernel_ms"], flush=True)
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
