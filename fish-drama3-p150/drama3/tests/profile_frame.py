# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: device time vs host time of one frame (traced backbone + 9 traced tower segments).

Replays each trace back to back with a sync (pure device time), then times one real frame's host pieces
(staging, readbacks, sampling, embedding rows) the way step_traced / decode do them.

    (fast env) python -m models.demos.blackhole.drama3.tests.profile_frame
"""
import json
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def main():
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")  # these checks open the card with the GDN L1_SMALL size
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.drama3.host.tower import CODEBOOK_TOP_K, RESIDUAL_CODEBOOK_SIZE, sample_top_k_top_p
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ref = str(VOICE / "ref_v2.wav")
    refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
        s = FastSession(mesh)
        rows = s.prompt(lines["fix"]["text"], refs)
        s._warm_T = rows.shape[0]
        s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.warmup(rows)
        tw, bb = s.tower, s.bb
        n = 50

        def dev(tid):
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(n):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            return (time.time() - t0) / n * 1e3

        res["backbone_trace_device_ms"] = dev(bb.trace_id)
        if s.device_sampling:  # DRAMA3_TOWER_DEVICE_SAMPLING=1: one fused trace per frame
            res["tower_fused_device_ms"] = dev(tw.fused_tid)
            print(json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
            return
        res["tower_segment_device_ms"] = [round(dev(t[0]), 3) for t in tw.traces]
        res["tower_device_ms"] = sum(res["tower_segment_device_ms"])

        # host pieces
        T = rows.shape[0]
        emb = s.embedder(rows[T - 1 : T])[0]
        t0 = time.time()
        for _ in range(n):
            bb._stage(emb, T)
        ttnn.synchronize_device(mesh)
        res["backbone_stage_ms"] = (time.time() - t0) / n * 1e3
        xn, lg = bb.dec_out
        t0 = time.time()
        for _ in range(n):
            ttnn.to_torch(ttnn.get_device_tensors(lg)[0])
        res["backbone_logits_readback_ms"] = (time.time() - t0) / n * 1e3
        _, _, lg0 = tw.traces[0]
        t0 = time.time()
        for _ in range(n):
            ttnn.to_torch(lg0)
        res["head_readback_ms"] = (time.time() - t0) / n * 1e3
        logits = ttnn.to_torch(lg0).float().reshape(-1)[:RESIDUAL_CODEBOOK_SIZE]
        full = torch.cat([logits, torch.full((4096 - RESIDUAL_CODEBOOK_SIZE,), float("-inf"))])
        g = torch.Generator().manual_seed(0)
        t0 = time.time()
        for _ in range(n):
            sample_top_k_top_p(full, 1.0, CODEBOOK_TOP_K, 0.9, g)
        res["head_sample_ms"] = (time.time() - t0) / n * 1e3
        cand = torch.randn(len(s.host.candidates))
        t0 = time.time()
        for _ in range(n):
            sample_top_k_top_p(cand, 1.0, len(s.host.candidates), 0.9, g, keep_crossing=True)
        res["main_sample_ms"] = (time.time() - t0) / n * 1e3
        lab = s.host.is_semantic(torch.tensor([s.host.sem0 + 5]))
        t0 = time.time()
        for _ in range(n):
            tw._put(tw.e_bufs[5], tw._emb_rows(tw.residual[1], torch.tensor([5]), lab))
        ttnn.synchronize_device(mesh)
        res["emb_put_ms"] = (time.time() - t0) / n * 1e3
        t0 = time.time()
        for _ in range(n):
            tw._stage_pos(T)
        ttnn.synchronize_device(mesh)
        res["tower_stage_pos_ms"] = (time.time() - t0) / n * 1e3
        res = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}
        print(json.dumps(res), flush=True)
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
