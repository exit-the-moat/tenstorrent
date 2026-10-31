# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare generated frame files: identical or the length of the identical prefix.

    python compare_frames.py <a.frames.pt> <b.frames.pt> [<a2> <b2> ...]
"""
import sys

import torch


def main():
    args = sys.argv[1:]
    for a, b in zip(args[::2], args[1::2]):
        x, y = torch.load(a), torch.load(b)
        n = min(len(x), len(y))
        same = next((i for i in range(n) if not torch.equal(x[i], y[i])), n)
        tag = "IDENTICAL" if same == len(x) == len(y) else f"prefix {same}"
        print(f"{tag:12s} frames {len(x)} vs {len(y)}  {a}  {b}", flush=True)


if __name__ == "__main__":
    main()
