# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 bb12: one real decode step (eager, per-layer outputs) with the backbone prefetch routing off / on in the SAME
process (same global CB, same L1 layout): where do they first differ? Random 128-row prompt, deterministic.

    EXTRA_ENV="DRAMA3_BB_PREFETCH=1 ..." run_check.sh <log> tests.check_bb_prefetch2
"""
import json
import os

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
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), weight_map(MODEL), cand, log=print)
        pf = bb.pf
        g = torch.Generator().manual_seed(1)
        E = torch.randn(128, args.dim, generator=g) * 0.05
        steps = [torch.randn(args.dim, generator=g) * 0.05 for _ in range(3)]

        def run(on, collect):
            bb.pf = pf if on else None
            bb.reset()
            bb.prefill(E)
            out = []
            for j, e in enumerate(steps):
                h, lg, per = bb.decode_eager(e, 128 + j, collect=collect)
                out.append((h, lg, per))
            bb.pf = pf
            return out

        def state_after_step0(on):
            bb.pf = pf if on else None
            bb.reset()
            bb.prefill(E)
            bb.decode_eager(steps[0], 128)
            bb.pf = pf
            st = {}
            for li, (dn, rec, conv) in enumerate(bb.gdn_state):
                st[f"gdn{li}.rec"] = ttnn.to_torch(rec).float()
                q = dn.split_conv_state
                rows = list(q) if q is not None else []
                for qi, t in enumerate(rows):
                    st[f"gdn{li}.conv{qi}"] = ttnn.to_torch(t).float()
            for ai, (k, v) in enumerate(bb.kv):
                st[f"kv{ai}.k"] = ttnn.to_torch(k).float()[:4]
                st[f"kv{ai}.v"] = ttnn.to_torch(v).float()[:4]
            return st

        res = {}
        nsteps = int(os.environ.get("BB12_STEPS", "6"))
        steps = [torch.randn(args.dim, generator=g) * 0.05 for _ in range(nsteps)]

        def lg(on):
            bb.pf = pf if on else None
            bb.reset()
            bb.prefill(E)
            out = [bb.decode_eager(e, 128 + j)[1] for j, e in enumerate(steps)]
            bb.pf = pf
            return out

        def cmp(a, b):
            return [int(not torch.equal(x, y)) for x, y in zip(a, b)]

        A0 = lg(False)  # no global CB yet
        pf.pf.start()
        A1 = lg(False)  # global CB allocated, prefetcher running idle
        B = lg(True)
        A2 = lg(False)
        B2 = lg(True)
        res = dict(gcb_vs_nogcb=cmp(A0, A1), on_vs_off=cmp(A1, B), off_after_on=cmp(A1, A2), on_repeat=cmp(B, B2))
        print("ab", json.dumps(res), flush=True)
        json.dump(res, open("/opt/ep0/port10/bb12/check_bb_prefetch2.json", "w"), indent=1)
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
