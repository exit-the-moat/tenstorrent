# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: DeviceSampler vs Fish's host sampler on the same logits + noise (codebook heads: top-k 30, top-p 0.9).

    python -m models.demos.blackhole.drama3.tests.check_device_sampler [--n 200] [--debug]
"""
import argparse
import json

import torch

import ttnn


def host_draw(x, noise, k=30, top_p=0.9):
    srt, ids = torch.sort(x, descending=True)
    cum = torch.cumsum(torch.softmax(srt, -1), -1)
    drop = (cum > top_p) | (torch.arange(srt.numel()) >= k)
    drop[0] = False
    probs = torch.softmax(srt.masked_fill(drop, float("-inf")), -1)
    return int(ids[torch.argmax(probs / noise)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.tt.device_sampler import K, DeviceSampler

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        s = DeviceSampler(mesh)
        g = torch.Generator().manual_seed(0)
        same = 0
        for i in range(a.n):
            x = torch.randn(1024, generator=g) * (1 + 3 * torch.rand(1, generator=g))
            noise = torch.empty(4096).exponential_(1, generator=g)
            xd = ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            nd = ttnn.from_torch(noise[:K].reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            code, onehot = s(xd, nd)
            c = int(round(float(ttnn.to_torch(code).reshape(-1)[0])))
            oh = ttnn.to_torch(onehot).float().reshape(-1)[:1024]
            h = host_draw(torch.cat([x, torch.full((3072,), float("-inf"))]), noise)
            same += int(c == h)
            if a.debug and i < 3:
                vals, idx = ttnn.topk(xd, K, dim=-1, largest=True, sorted=True)
                print(json.dumps(dict(
                    host=h, dev=c, onehot_sum=float(oh.sum()), onehot_arg=int(oh.argmax()),
                    topk_vals=[round(v, 3) for v in ttnn.to_torch(vals).float().reshape(-1)[:6].tolist()],
                    topk_idx=ttnn.to_torch(idx).reshape(-1)[:6].tolist(),
                    host_top=torch.sort(x, descending=True).indices[:6].tolist(),
                )), flush=True)
        print(json.dumps(dict(n=a.n, agree=same / a.n)), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
