# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 bb12: DRAMA3_BB_PREFETCH=1 checks on the real backbone. (1) every receiver-contiguous copy holds the source
weight values (pad = 0); (2) each prefetched matmul on a random row vs the default matmul on the same row.

    EXTRA_ENV="DRAMA3_BB_PREFETCH=1 ..." run_check.sh <log> tests.check_bb_prefetch
"""
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)


def main():
    from models.demos.blackhole.drama3.host.generate import MODEL
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    cfg = json.load(open(f"{MODEL}/config.json"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=64 << 20)
    res = {}
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), weight_map(MODEL), cand, log=print)
        g = bb.pf
        bad = []
        for e in g.order:
            s = ttnn.to_torch(e.src).float().reshape(e.K, -1)
            w = ttnn.to_torch(e.w).float().reshape(e.K, -1)
            n = s.shape[-1]
            d = int((w[:, :n] != s).sum())
            if d or float(w[:, n:].abs().max() if w.shape[-1] > n else 0) != 0:
                bad.append((e.name, d))
        res["copy_mismatch"] = bad
        print("copies", len(g.order), "mismatching:", bad[:10], flush=True)
        # matmuls, one random row, in group order through a frame (exactly the decode-body call pattern)
        torch.manual_seed(0)
        xs = {}
        refs = []
        L1 = ttnn.L1_MEMORY_CONFIG
        ckc = bb.layers[0].attention.compute_kernel_config_decode
        for e in g.order:
            if e.K not in xs:
                xs[e.K] = ttnn.from_torch(torch.randn(1, 1, e.K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=L1)
            refs.append(ttnn.to_torch(ttnn.linear(xs[e.K], e.src, compute_kernel_config=ckc, memory_config=L1)).float().reshape(-1))
        outs = []
        with g.frame():
            for e in g.order:
                outs.append(ttnn.linear(xs[e.K], e.src, compute_kernel_config=ckc, memory_config=L1))
        md = []
        for e, r, o in zip(g.order, refs, outs):
            y = ttnn.to_torch(o).float().reshape(-1)[: r.numel()]
            if not torch.equal(y, r):
                md.append((e.name, int((y != r).sum()), float((y - r).abs().max())))
        res["matmul_mismatch"] = md
        print("matmuls mismatching:", md[:10], flush=True)
        json.dump(res, open("/opt/ep0/port10/bb12/check_bb_prefetch.json", "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
