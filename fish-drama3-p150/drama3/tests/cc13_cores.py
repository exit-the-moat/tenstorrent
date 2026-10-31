# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 cc13: which Tensix cores each traced piece of a frame uses (backbone decode, embed, fused tower, the
card codec quantizer + decoder windows), from the device profiler (TT_METAL_DEVICE_PROFILER=1,
TT_METAL_PROFILER_DIR=<dir>). Each trace is replayed once, 30 ms apart; the log is split at gaps > 5 ms.

    (--interactive env) run_check.sh <log> tests.cc13_cores --out cores.json
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV
from models.demos.blackhole.drama3.tests.profile_frame7 import session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port10/cc13/cores.json")
    ap.add_argument("--tower-mlp", default="bfp8")
    a = ap.parse_args()
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=256 << 20)
    g = mesh.compute_with_storage_grid_size()
    v2l = {}
    for x in range(g.x):
        for y in range(g.y):
            v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            v2l[(v.x, v.y)] = (x, y)
    s, rows = session(mesh, "close", a.tower_mlp)
    cc = s.card_codec
    pieces = [("backbone", s.bb.trace_id), ("tower", s.tower.fused_tid)]
    if getattr(s.bb, "embed_tid", None) is not None:
        pieces.append(("embed", s.bb.embed_tid))
    for w, (inp, tid, out) in sorted(cc.dq.traces.items()):
        pieces.append((f"quant{w // 4 if w > 96 else w}", tid))
    for w, (tid, out, L) in sorted(getattr(cc, "dev_traces", {}).items()):
        pieces.append((f"codec{w}", tid))
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    for name, tid in pieces:
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        time.sleep(0.03)
        ttnn.ReadDeviceProfiler(mesh)
    ttnn.close_device(mesh)
    log = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs/profile_log_device.csv"
    ev = []  # (t, rid, (x, y))
    with open(log) as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        for r in csv.DictReader(f, skipinitialspace=True):
            if r["zone name"].endswith("-KERNEL") and r["type"] == "ZONE_START":
                ev.append((int(r["time[cycles since reset]"]), int(r["run host ID"]), (int(r["core_x"]), int(r["core_y"]))))
    ev.sort()
    groups, cur = [], []
    for e in ev:
        if cur and (e[0] - cur[-1][0]) / freq > 5000:
            groups.append(cur)
            cur = []
        cur.append(e)
    groups.append(cur)
    groups = groups[-len(pieces):]  # the warm-up programs before the first read come first
    res = dict(grid=[g.x, g.y], pieces={})
    for (name, _), evs in zip(pieces, groups):
        prog = defaultdict(set)
        for t, rid, c in evs:
            prog[rid].add(v2l.get(c, ("v",) + c))
        allc = set().union(*prog.values())
        far = {rid: sorted(cs) for rid, cs in prog.items() if any(isinstance(c[0], int) and (c[0] >= 8 or c[1] >= 8) for c in cs)}
        span = (evs[-1][0] - evs[0][0]) / freq
        res["pieces"][name] = dict(
            n_programs=len(prog), span_us=round(span, 1), n_cores=len(allc),
            max_x=max(c[0] for c in allc if isinstance(c[0], int)), max_y=max(c[1] for c in allc if isinstance(c[0], int)),
            n_far_programs=len(far), far_example={str(k): v for k, v in list(far.items())[:3]},
            cols={str(x): sum(1 for p in prog.values() if any(c[0] == x for c in p)) for x in range(g.x)},
            rows={str(y): sum(1 for p in prog.values() if any(c[1] == y for c in p)) for y in range(g.y)},
            unmapped=sorted(str(c) for c in allc if not isinstance(c[0], int))[:5],
        )
        print(name, json.dumps({k: v for k, v in res["pieces"][name].items() if k != "far_example"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
