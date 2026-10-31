# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: K-block sweep (QWEN36_MOE_DECODE_BLOCK_GATE_UP / _DOWN) of the indexed MoE decode: the backbone decode
trace is re-captured per setting and replayed; candidate logits compared with the first setting."""
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def main():
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    cfg = json.load(open(f"{MODEL}/config.json"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=512 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), weight_map(MODEL), cand)
        emb = torch.randn(args.dim) * 0.3
        ref = None
        for gu, dn in ((32, 8), (16, 8), (64, 8), (32, 4), (32, 16), (64, 16), (16, 4)):
            os.environ["QWEN36_MOE_DECODE_BLOCK_GATE_UP"], os.environ["QWEN36_MOE_DECODE_BLOCK_DOWN"] = str(gu), str(dn)
            try:
                bb.reset()
                bb.capture(pos=700)
                bb.reset()
                bb.decode(emb, 700)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(30):
                    ttnn.execute_trace(mesh, bb.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.time() - t0) / 30 * 1e3
                bb.reset()
                _, lg = bb.decode(emb, 700)
                if ref is None:
                    ref = lg
                d = float((lg - ref).abs().max())
                print(json.dumps(dict(gate_up=gu, down=dn, ms=round(ms, 3), max_logit_diff_vs_first=d)), flush=True)
                ttnn.release_trace(mesh, bb.trace_id)
            except Exception as e:  # noqa: BLE001
                print(json.dumps(dict(gate_up=gu, down=dn, error=str(e)[:150])), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
