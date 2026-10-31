# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: codec window decode on the CPU, 96-frame window (the steady-state Fish window): torch.compile dynamic
(the pipeline today) vs static shapes, per thread count, no other load. ms per decoded frame + max |diff|.

    PYTHONPATH=$PYTHONPATH:/opt/ep0/pyext-codec python -m models.demos.blackhole.drama3.tests.bench_codec2 <eager|dynamic|static> 4,6,8
"""
import json
import os
import sys
import time

import torch

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/opt/ep0/inductor_cache")
from models.demos.blackhole.drama3.host.codec import load_codec  # noqa: E402

c = load_codec()
c.fold_weight_norm()
codes = torch.load("/opt/ep0/day4b/gate8/fix.frames.pt")[:96, 1:].T.contiguous()[None]
mode = sys.argv[1]
threads = [int(t) for t in sys.argv[2].split(",")]
with torch.inference_mode():
    z = c.quantizer.decode(codes)
    ref = c.decoder(z)
    if mode == "eager":
        f = c.decoder
    elif mode == "dynamic":
        f = torch.compile(c.decoder, dynamic=True)
    else:
        f = torch.compile(c.decoder, dynamic=False, mode=None if mode == "static" else mode)
    t0 = time.time()
    f(z)
    comp = time.time() - t0
    for th in threads:
        torch.set_num_threads(th)
        f(z)
        ts = []
        for _ in range(3):
            t0 = time.time()
            out = f(z)
            ts.append(time.time() - t0)
        tq = []
        for _ in range(3):
            t0 = time.time()
            c.quantizer.decode(codes)
            tq.append(time.time() - t0)
        print(json.dumps(dict(mode=mode, threads=th, compile_s=round(comp, 1), decoder_ms_per_frame=round(min(ts) / 96 * 1e3, 2),
                              quantizer_ms_per_frame=round(min(tq) / 96 * 1e3, 2), max_diff=float((out - ref).abs().max()))), flush=True)
