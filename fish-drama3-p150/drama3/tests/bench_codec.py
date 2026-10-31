# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: codec decode speed on the host CPU (full clip and Fish's 96-frame stream window), with and without
fold_weight_norm, for a few thread counts; plus the max |diff| folded vs unfolded.

    PYTHONPATH=$PYTHONPATH:/opt/ep0/pyext-codec python -m models.demos.blackhole.drama3.tests.bench_codec
"""
import json
import time

import torch

from models.demos.blackhole.drama3.host import codec as C


def bench(fn, n=3):
    fn()
    t = []
    for _ in range(n):
        t0 = time.time()
        fn()
        t.append(time.time() - t0)
    return min(t)


def main():
    codes = torch.load("/opt/ep0/day3/out/fix.frames.pt")[:, 1:].T.contiguous()  # [10, 254]
    codec = C.load_codec()
    win = codes[:, :96]
    with torch.inference_mode():
        ref = codec.from_indices(codes[None])
        for th in (8, 16):
            torch.set_num_threads(th)
            r = dict(threads=th, folded=False)
            r["full_254_frames_s"] = round(bench(lambda: codec.from_indices(codes[None]), 2), 3)
            r["window_96_frames_s"] = round(bench(lambda: codec.from_indices(win[None])), 3)
            print(json.dumps(r), flush=True)
        codec.fold_weight_norm()
        out = codec.from_indices(codes[None])
        print(json.dumps(dict(fold_max_abs_diff=float((out - ref).abs().max()), ref_peak=float(ref.abs().max()))), flush=True)
        for th in (4, 8, 16):
            torch.set_num_threads(th)
            r = dict(threads=th, folded=True)
            r["full_254_frames_s"] = round(bench(lambda: codec.from_indices(codes[None]), 2), 3)
            r["window_96_frames_s"] = round(bench(lambda: codec.from_indices(win[None])), 3)
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
