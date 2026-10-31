# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: FLOPs and parameters of the codec decode (quantizer.decode + decoder) per frame, CPU, for the card-codec estimate."""
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from models.demos.blackhole.drama3.host.codec import load_codec

c = load_codec()
c.fold_weight_norm()
n = 96
codes = torch.zeros(1, 10, n, dtype=torch.long)
with torch.inference_mode():
    z = c.quantizer.decode(codes)
    fq = FlopCounterMode(display=False)
    with fq:
        c.quantizer.decode(codes)
    fd = FlopCounterMode(display=False)
    with fd:
        c.decoder(z)
    torch.set_num_threads(6)
    c.decoder(z)
    t0 = time.time()
    c.decoder(c.quantizer.decode(codes))
    eager_s = time.time() - t0
pq = sum(p.numel() for p in c.quantizer.parameters())
pd = sum(p.numel() for p in c.decoder.parameters())
print(json.dumps(dict(window_frames=n, quantizer_gflop_per_frame=fq.get_total_flops() / n / 1e9, decoder_gflop_per_frame=fd.get_total_flops() / n / 1e9,
                      quantizer_params_m=pq / 1e6, decoder_params_m=pd / 1e6, eager_6threads_ms_per_frame=eager_s / n * 1e3)))
