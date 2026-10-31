# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track A: what do the 40 per-layer routing readbacks of the grouped-MoE suffix prefill cost? Times the
suffix prefill (backbone + tower, synced at the end only, as in the pipeline) three ways:
  normal  : as is (each MoE layer reads its routing to the host, builds the index lists, uploads them)
  replay  : the routing of a recorded run is fed back from host memory: no device->host sync, same host work
  host    : the host index build alone on the recorded routing (no device)
replay's numbers are wrong on purpose (stale routing); only its time matters: it bounds a device-side grouping.

    (interactive env) python -m models.demos.blackhole.drama3.tests.profile_suffix_sync [--line fix] [--n 3]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="fix")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    import models.demos.blackhole.qwen36.tt.moe.experts as ex_mod
    import models.demos.blackhole.qwen36.tt.moe.prefill_grouped as pg

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    refs = [(Path(a.ref).with_suffix(".txt").read_text().strip(), reference_codes(a.ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
        s = FastSession(mesh)
        rows = s.prompt(lines[a.line]["text"], refs)
        P = s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.set_voice(rows)
        res["suffix_rows"] = int(rows.shape[0] - P)

        def timed(tag):
            ts = []
            for _ in range(a.n):
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                s._prefill_line(rows)
                ttnn.synchronize_device(mesh)
                ts.append(round(time.time() - t0, 4))
            res[tag] = ts
            print(tag, ts, flush=True)

        timed("warm")
        timed("normal")
        orig = ex_mod.grouped_prefill_forward
        rec = []

        def recording(h, r, w, c, m, valid_len=None, force_M=None):
            S, E = h.shape[-2], c.num_experts
            rec.append((ttnn.to_torch(r).float().reshape(S, E), valid_len))
            return orig(h, r, w, c, m, valid_len=valid_len, force_M=force_M)

        ex_mod.grouped_prefill_forward = recording
        s._prefill_line(rows)
        ttnn.synchronize_device(mesh)
        it = {"i": 0}

        def replay(h, r, w, c, m, valid_len=None, force_M=None):
            rr, n = rec[it["i"] % len(rec)]
            it["i"] += 1
            return pg._grouped(h, rr, torch.arange(n), w, c, m, force_M)

        ex_mod.grouped_prefill_forward = replay
        timed("replay")
        ex_mod.grouped_prefill_forward = orig
        # host index build alone (the torch part of _grouped), 40 layers
        t0 = time.time()
        for _ in range(a.n):
            for rr, n in rec:
                rows_ = torch.arange(n)
                K, E = 8, rr.shape[-1]
                w, e = torch.topk(rr[rows_], K, dim=-1)
                fe = e.reshape(-1)
                order = torch.argsort(fe, stable=True)
                counts = torch.bincount(fe, minlength=E)
                M = next(m for m in pg.M_BUCKETS if m >= int(counts.max()))
                starts = torch.cumsum(counts, 0) - counts
                slot = torch.empty_like(fe)
                slot[order] = torch.arange(fe.numel()) - starts[fe[order]]
                dest = fe * M + slot
                gi = torch.zeros(E * M, dtype=torch.long)
                gi[dest] = rows_.repeat_interleave(K)
                sw = torch.zeros(E * M)
                sw[dest] = w.reshape(-1)
        res["host_build_s_per_prefill"] = round((time.time() - t0) / a.n, 4)
        res["layers"] = len(rec)
        res["max_expert_rows"] = [int(torch.bincount(rr[:n].topk(8, -1).indices.reshape(-1), minlength=rr.shape[-1]).max()) for rr, n in rec]
        print(json.dumps(res), flush=True)
        if a.out:
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
