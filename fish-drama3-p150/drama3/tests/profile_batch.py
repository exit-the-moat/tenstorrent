# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 5b: device time of the batched decode pieces vs the batch size (traced replays, back to back).

    DRAMA3_CARD_CODEC=0 run_check.sh <log> tests.profile_batch --sizes 1,2,4,8
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn


def _dense(fn):
    """The dense 256-slot MoE decode (union of the rows' experts) instead of the batched indexed one."""
    os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "0"
    try:
        return fn()
    finally:
        os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="/opt/ep0/day5b/profile_batch.json")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    os.environ["DRAMA3_KV_STREAMS"] = str(max(sizes))
    os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"
    os.environ["DRAMA3_CARD_CODEC"] = "0"
    from models.demos.blackhole.drama3.host.batch import BatchState
    from models.demos.blackhole.drama3.host.generate import TT_ENV, log, reference_codes
    from models.demos.blackhole.drama3.host.generate_fast import FastSession

    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    ref = "/opt/ep0/voice/ref_v2.wav"
    codes = reference_codes(ref, "/opt/ep0/day3")
    ref_text = Path(ref).with_suffix(".txt").read_text().strip()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=256 << 20)
    res = {}
    try:
        s = FastSession(mesh)
        rows = s.prompt("So, did we exit the moat? Not yet.", [(ref_text, codes)])
        s._warm_T = rows.shape[0]
        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + [rows])
        s.warmup(rows, capture=False)
        bb = s.bb
        gdn_layer = next(l for l in bb.layers if not l.is_full_attention)
        att_layer = next(l for l in bb.layers if l.is_full_attention)
        for B in sizes:
            st = BatchState(s, B)
            st.stage(torch.randn(B, bb.args.dim) * 0.3, torch.full((B,), rows.shape[0], dtype=torch.long))
            st.batch_mode(True)
            pieces = {
                "backbone": lambda: st._bb_body(),
                "gdn_mixer": lambda: gdn_layer.attention.forward(st.x_buf, mode="recurrent"),
                "attn_mixer": lambda: att_layer.attention.forward(
                    st.x_buf, st.cos_buf, st.sin_buf, position_tensor=st.pos_buf, page_table=st.table
                ),
                "moe": lambda: gdn_layer.feed_forward.forward(st.x_buf, mode="decode"),
                "norm": lambda: gdn_layer.attention_norm(st.x_buf, mode=None),
                "moe_dense": lambda: _dense(lambda: gdn_layer.feed_forward.forward(st.x_buf, mode="decode")),
                "tower": lambda: st._tw_body(st.x_buf),
            }
            r = {}
            for name, fn in pieces.items():
                try:
                    fn()  # compile
                    ttnn.synchronize_device(mesh)
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    fn()
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    t0 = time.time()
                    for _ in range(a.reps):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    r[name] = round((time.time() - t0) / a.reps * 1e3, 3)
                    ttnn.release_trace(mesh, tid)
                except Exception as e:
                    r[name] = f"FAILED {str(e)[:200]}"
                log(f"B={B} {name}: {r[name]} ms")
            st.batch_mode(False)
            res[B] = r
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
