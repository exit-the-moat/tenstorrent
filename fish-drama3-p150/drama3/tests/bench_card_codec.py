# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: where does a card-codec window go (host prep, upload, device, readback)?"""
import json
import time

import numpy as np
import torch

import ttnn
from models.demos.blackhole.drama3.host.stream_codec import CardCodec


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        cc = CardCodec(mesh)
        dd = cc.dec
        z = np.random.randn(4 * 64, 1024).astype(np.float32) * 0.5
        for rep in range(3):
            t = {}
            t0 = time.time()
            zz = torch.zeros(1, 4 * cc.WINDOW, 1024)
            zz[0, : z.shape[0]] = torch.from_numpy(z)
            t["host_pad_ms"] = (time.time() - t0) * 1e3
            t0 = time.time()
            zd = ttnn.from_torch(zz, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            ttnn.synchronize_device(mesh)
            t["upload_ms"] = (time.time() - t0) * 1e3
            t0 = time.time()
            x, _ = dd._body(zd)
            t["enqueue_ms"] = (time.time() - t0) * 1e3
            ttnn.synchronize_device(mesh)
            t["device_total_ms"] = (time.time() - t0) * 1e3
            t0 = time.time()
            a = ttnn.to_torch(x).float()[0, : dd.L_out, 0]
            t["readback_ms"] = (time.time() - t0) * 1e3
            print(json.dumps({k: round(v, 1) for k, v in t.items()}), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
