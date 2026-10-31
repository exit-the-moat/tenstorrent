# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 5b: teacher-forced backbone decode, batch of B identical streams vs the one-row batch (= the solo
pipeline's math, frames bit-identical): candidate-logit PCC, max |diff| and top-1 per step, and whether the B rows
agree with each other.

    DRAMA3_CARD_CODEC=0 run_check.sh <log> tests.check_batch_numerics --sizes 2,4,8 --frames /opt/ep0/day4b/out_v2/close.frames.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,4,8")
    ap.add_argument("--frames", default="/opt/ep0/day4b/out_v2/close.frames.pt")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="/opt/ep0/day5b/check_batch_numerics.json")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    os.environ["DRAMA3_KV_STREAMS"] = str(max(sizes))
    os.environ["QWEN36_MOE_DECODE_INDEXED_BATCH"] = "1"
    os.environ["DRAMA3_CARD_CODEC"] = "0"
    import yaml

    from models.demos.blackhole.drama3.host.batch import BatchState
    from models.demos.blackhole.drama3.host.generate import TT_ENV, log, reference_codes
    from models.demos.blackhole.drama3.host.generate_fast import FastSession

    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    ref = "/opt/ep0/voice/ref_v2.wav"
    codes = reference_codes(ref, "/opt/ep0/day3")
    ref_text = Path(ref).with_suffix(".txt").read_text().strip()
    text = {l["id"]: l for l in yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]}["close"]["text"]
    frames = torch.load(a.frames)[: a.steps]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=256 << 20)
    res = {}
    try:
        s = FastSession(mesh)
        rows = s.prompt(text, [(ref_text, codes)])
        s._warm_T = rows.shape[0]
        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + [rows])
        s.warmup(rows, capture=False)
        T = rows.shape[0]
        embs = s.embedder(frames)  # teacher-forced inputs
        outs = {}
        for B in [1] + sizes:
            st = BatchState(s, B)
            st.capture()
            for b in range(B):  # every stream: the same line, prefilled in its own page-table row
                st.stream_table(b)
                if b == 0:
                    st.stream_table(0)
                    s.set_voice(rows)
                s._prefill_line(rows)
                if b == 0:
                    state = st.read_gdn()
            st.stream_table(None)
            st.load_gdn([state] * B)
            lg = []
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for i in range(frames.shape[0]):
                st.stage(embs[i : i + 1].expand(B, -1).contiguous(), torch.full((B,), T + i, dtype=torch.long))
                ttnn.execute_trace(mesh, st.bb_tid, cq_id=0, blocking=False)
                lg.append(ttnn.to_torch(st.dec_out[1]).float().reshape(B, -1)[:, : s.bb.n_cand].clone())
            bb_ms = 1e3 * (time.time() - t0) / frames.shape[0]  # port6: backbone trace + stage + logits read per step
            outs[B] = torch.stack(lg)  # [steps, B, n_cand]
            res.setdefault("backbone_ms_per_step", {})[B] = round(bb_ms, 2)
            log(f"B={B}: backbone {bb_ms:.2f} ms per step (stage + trace + logits read)")
            ttnn.release_trace(mesh, st.bb_tid)
            ttnn.release_trace(mesh, st.tw_tid)
            if B == 1:
                continue
            ref1 = outs[1][:, 0]
            r = {}
            for b in range(B):
                x = outs[B][:, b]
                r[f"row{b}"] = dict(
                    pcc_min=round(min(pcc(x[i], ref1[i]) for i in range(len(x))), 7),
                    maxdiff=round(float((x - ref1).abs().max()), 5),
                    top1=f"{sum(int(x[i].argmax()) == int(ref1[i].argmax()) for i in range(len(x)))}/{len(x)}",
                    identical=bool(torch.equal(x, ref1)),
                )
            r["rows_equal_each_other"] = all(torch.equal(outs[B][:, b], outs[B][:, 0]) for b in range(B))
            res[B] = r
            log(f"B={B}: {json.dumps(r)}")
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
