# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: device time per card-codec decoder op (sync after each), one 96-frame window."""
import json
import time
from collections import defaultdict

import torch

import ttnn
from models.demos.blackhole.drama3.host.stream_codec import CardCodec


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        cc = CardCodec(mesh)
        dd = cc.dec
        z = ttnn.from_torch(torch.randn(1, dd.L0, 1024) * 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        agg = defaultdict(float)
        for rep in range(2):
            x, L = z, dd.L0
            ttnn.synchronize_device(mesh)
            for si, (kind, p) in enumerate(dd.stages):
                t0 = time.time()
                if kind == "conv":
                    x = dd._conv(x, L, p)
                elif kind == "snake":
                    x = dd._snake_op(x, p)
                elif kind == "trans":
                    c, s, cout = p
                    y = dd._conv(x, L, c)
                    ttnn.synchronize_device(mesh)
                    t1 = time.time()
                    if rep:
                        agg[f"trans_conv L={L} {c.cin}->{c.cout}"] += (t1 - t0) * 1e3
                    y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
                    y = ttnn.reshape(y, (1, L * s, cout))
                    L = L * s
                    x = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
                    ttnn.synchronize_device(mesh)
                    if rep:
                        agg[f"trans_relayout L={L} C={cout}"] += (time.time() - t1) * 1e3
                    continue
                elif kind == "res":
                    s1, c7, s2, c1 = p
                    tt = {}
                    t1 = time.time(); y = dd._snake_op(x, s1); ttnn.synchronize_device(mesh); tt["snake"] = time.time() - t1
                    t1 = time.time(); y = dd._conv(y, L, c7); ttnn.synchronize_device(mesh); tt["conv7"] = time.time() - t1
                    t1 = time.time(); y = dd._snake_op(y, s2); ttnn.synchronize_device(mesh); tt["snake"] += time.time() - t1
                    t1 = time.time(); y = dd._conv(y, L, c1); ttnn.synchronize_device(mesh); tt["conv1"] = time.time() - t1
                    t1 = time.time(); x = ttnn.add(x, y); ttnn.synchronize_device(mesh); tt["add"] = time.time() - t1
                    if rep:
                        for k, v in tt.items():
                            agg[f"res_{k} L={L} C={c7.cin}"] += v * 1e3
                    continue
                elif kind == "tanh":
                    x = ttnn.tanh(x)
                ttnn.synchronize_device(mesh)
                if rep:
                    agg[f"{kind} L={L}"] += (time.time() - t0) * 1e3
        tot = sum(agg.values())
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"{v:7.2f} ms  {k}", flush=True)
        print(f"total {tot:.1f} ms (synced per op)", flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
