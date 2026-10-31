# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: the line-suffix prefill's 128-row ttnn.linear / ttnn.matmul calls (backbone + tower): default program vs
core_grid-picked programs, accepted only when bit-identical (max |diff| == 0) and faster."""
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes

REC = dict(on=False, calls={}, n={})


def main():
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        s = FastSession(mesh)
        lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
        ref = str(VOICE / "ref_v2.wav")
        refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
        rows = s.prompt(lines["fix"]["text"], refs)
        s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.set_voice(rows)
        for name in ("linear", "matmul"):
            fn = getattr(ttnn, name)

            def w(*args, _fn=fn, _name=name, **kw):
                if REC["on"] and len(args) >= 2 and isinstance(args[1], ttnn.Tensor) and args[0].shape[-2] == 128 and "program_config" not in kw:
                    key = json.dumps([_name, list(args[0].shape), list(args[1].shape), str(args[1].dtype), str(kw.get("dtype")), str(kw.get("activation"))])
                    REC["n"][key] = REC["n"].get(key, 0) + 1
                    if key not in REC["calls"]:
                        REC["calls"][key] = (_fn, ttnn.clone(args[0]), args[1], dict(kw))
                return _fn(*args, **kw)

            setattr(ttnn, name, w)
        REC["on"] = True
        s._prefill_line(rows)
        REC["on"] = False

        def timed(f):
            out = f()
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(5):
                ttnn.deallocate(f())
            ttnn.synchronize_device(mesh)
            return (time.time() - t0) / 5 * 1e3, ttnn.to_torch(out).float()

        tot = 0.0
        for key, (fn, x, wt, kw) in REC["calls"].items():
            d_ms, ref_o = timed(lambda: fn(x, wt, **kw))
            best = None
            for gy, gx in ((10, 11), (8, 8), (8, 4), (4, 8)):
                try:
                    ms, o = timed(lambda: fn(x, wt, core_grid=ttnn.CoreGrid(y=gy, x=gx), **kw))
                except Exception:  # noqa: BLE001
                    continue
                if torch.equal(o, ref_o) and ms < d_ms * 0.95 and (best is None or ms < best[0]):
                    best = (round(ms, 3), f"{gx}x{gy}")
            save = (d_ms - best[0]) * REC["n"][key] if best else 0.0
            tot += save
            print(json.dumps(dict(sig=json.loads(key), calls=REC["n"][key], default_ms=round(d_ms, 3), best=best, saved_ms=round(save, 2))), flush=True)
        print("TOTAL saved ms per suffix prefill (bit-identical):", round(tot, 1), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
