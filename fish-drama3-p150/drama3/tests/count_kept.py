# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7: how many ranks does the main-token draw keep (top-p 0.9 with the crossing kept, no top-k)? Runs
host.generate_fast unchanged, with a counter around tower_device.sample_with_noise (main draws only: the RAS
draw has top-k 30). Writes the histogram of kept ranks to --kept-out."""
import json
import sys

import torch

from models.demos.blackhole.drama3.tt import tower_device

KEPT = []
_orig = tower_device.sample_with_noise


def counted(logits, temperature, top_k, top_p, noise, keep_crossing=False, m=64):
    if top_k > 1000:  # the main draw (top_k = all candidates)
        scaled = logits.float() / max(temperature, 1e-5)
        p = torch.softmax(scaled, -1).sort(descending=True).values
        excl = torch.cumsum(p, -1) - p
        KEPT.append(int((excl <= top_p).sum()))
    return _orig(logits, temperature, top_k, top_p, noise, keep_crossing, m)


tower_device.sample_with_noise = counted

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--kept-out") + 1]
    i = sys.argv.index("--kept-out")
    del sys.argv[i : i + 2]
    from models.demos.blackhole.drama3.host import generate_fast

    try:
        generate_fast.main()
    finally:
        k = torch.tensor(KEPT)
        res = dict(n=len(KEPT), max=int(k.max()) if len(KEPT) else None,
                   over={str(t): int((k > t).sum()) for t in (16, 32, 64, 128, 256, 512)},
                   quantiles={str(q): float(k.float().quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)} if len(KEPT) else None)
        json.dump(res, open(out, "w"), indent=1)
        print("KEPT", json.dumps(res), flush=True)
