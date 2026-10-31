# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b debug: DeviceSampler with the noise as a row slice of a [1,1,9,32] buffer, eager and traced."""
import json

import torch

import ttnn
from models.demos.blackhole.drama3.tests.check_device_sampler import host_draw


def main():
    from models.demos.blackhole.drama3.tt.device_sampler import K, DeviceSampler

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    try:
        s = DeviceSampler(mesh)
        g = torch.Generator().manual_seed(0)
        xbuf = ttnn.from_torch(torch.zeros(1, 1, 1, 1024), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        nbuf = ttnn.from_torch(torch.ones(1, 1, 9, K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)

        def body():
            out = []
            for r in range(9):
                nz = ttnn.slice(nbuf, [0, 0, r, 0], [1, 1, r + 1, K])
                out.append(s(xbuf, nz)[0])
            return ttnn.concat(out, dim=-1)

        def load():
            x = torch.randn(1024, generator=g) * 3
            noise = torch.stack([torch.empty(4096).exponential_(1, generator=g) for _ in range(9)])
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), xbuf)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(noise[None, None, :, :K].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), nbuf)
            xx = torch.cat([x, torch.full((3072,), float("-inf"))])
            return [host_draw(xx, noise[r]) for r in range(9)]

        h = load()
        d = ttnn.to_torch(body()).float().reshape(-1)[:9].round().long().tolist()
        print(json.dumps(dict(mode="eager_slice", host=h, dev=d)), flush=True)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = body()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        agree = 0
        for i in range(20):
            h = load()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            d = ttnn.to_torch(out).float().reshape(-1)[:9].round().long().tolist()
            agree += sum(int(a == b) for a, b in zip(h, d))
            if i < 2:
                print(json.dumps(dict(mode="traced", host=h, dev=d)), flush=True)
        print(json.dumps(dict(traced_agree=agree / 180)), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
