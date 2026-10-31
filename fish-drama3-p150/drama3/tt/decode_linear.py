# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tuned program configs for the one-token matmuls of the backbone decode and the tower (Day 4b,
default on, DRAMA3_DECODE_LINEAR=0 to disable).

ttnn.linear picks its own program for a [1, K] x [K, N] matmul; for the small and mid-sized decode weights
it uses few cores (e.g. the router 2048 x 256: 51 us, the GDN / attention o_proj 4096 x 2048: 84 us).
tests/tune_decode_linear.py swept MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast in0, per_core_N,
in0_block_w) per weight signature as one-op traces; the table below is its result (one-op device time
default -> tuned, calls per frame). install() wraps ttnn.linear / ttnn.matmul once: a call with a one-row input,
no program_config of its own, no activation or silu (moved into the config) and a weight signature in the table
gets the tuned config.
"""

import math
import os

import ttnn

ENABLED = os.environ.get("DRAMA3_DECODE_LINEAR", "1") == "1"

# (weight shape, weight dtype, output dtype) -> (per_core_N, in0_block_w, out_subblock_w)
TABLE = {
    ((1, 1, 2048, 1), "BFLOAT16", None): (1, 16, 1),  # shared-expert gate, 40/frame: 48.8 -> 8.4 us
    ((4096, 2048), "BFLOAT8_B", None): (3, 16, 1),  # GDN + attention o_proj, 30 + 10/frame: 84 -> 33 us
    ((1, 1, 2048, 256), "BFLOAT16", None): (1, 16, 1),  # router, 40/frame: 51 -> 13 us
    ((2048, 512), "BFLOAT4_B", None): (2, 16, 2),  # shared expert up/gate, 40/frame: 33 -> 14 us
    ((2048, 512), "BFLOAT8_B", None): (2, 16, 2),  # attention k / v, shared expert: 34-37 -> 14.5 us
    ((6144, 2048), "BFLOAT16", None): (3, 4, 1),  # tower w2, 13/frame: 122 -> 70 us
    ((2048, 2048), "BFLOAT16", None): (3, 4, 1),  # tower eh halves, 26/frame: 49 -> 30 us
    ((2048, 1024), "BFLOAT16", "FLOAT32"): (2, 16, 2),  # tower heads, 9/frame: 41 -> 21 us
    ((2048, 12352), "BFLOAT8_B", None): (12, 1, 4),  # GDN in-projection, 30/frame: 82 -> 79.5 us
    ((2048, 8192), "BFLOAT8_B", None): (12, 4, 4),  # attention q|gate: 59 -> 56.5 us
    # port5: the tower MLP in BFP8 (opt-in, run_gen --tower-mlp bfp8; tests/tune_decode_linear.py, port5/linear_tune_bfp8.json)
    ((2048, 9216), "BFLOAT8_B", None): (12, 4, 4),  # port5 DRAMA3_ATTN_QKV_MERGED: [q|gate | k | v] (as q|gate)
    ((6144, 2048), "BFLOAT8_B", None): (2, 8, 2),  # tower w2 BFP8, 13/frame: 122 -> 46 us (bf16 tuned: 70)
    ((2048, 6144), "BFLOAT8_B", None): (6, 4, 2),  # tower w1 / w3 BFP8, 26/frame: 61 -> 47 us (bf16: 75-81)
    # port8: the tower MLP in BFP4 (precision candidate, --tower-mlp bfp4): the BFP8 configs, not swept
    ((6144, 2048), "BFLOAT4_B", None): (2, 8, 2),
    ((2048, 6144), "BFLOAT4_B", None): (6, 4, 2),
}
_installed = False
# rows of a one-tile input that take the tuned programs (batch mode sets it to B; default: one row)
ROWS = 1


def _dtype(d):
    return None if d is None else str(d).split(".")[-1]


def _config(mesh, w, out_dtype, fused_activation=None):
    key = (tuple(w.shape), _dtype(w.dtype), _dtype(out_dtype))
    hit = TABLE.get(key)
    if hit is None:
        return None
    pn, ibw, sw = hit
    nt = math.ceil(w.shape[-1] / 32)
    g = mesh.compute_with_storage_grid_size()
    cores = math.ceil(nt / pn)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(min(g.x, cores), math.ceil(cores / g.x)),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pn,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def install(mesh):
    global _installed
    if _installed or not ENABLED:
        return
    _installed = True
    cache = {}
    for name in ("linear", "matmul"):
        fn = getattr(ttnn, name)

        def wrapped(*args, _fn=fn, **kw):
            act = kw.get("activation")
            if (
                len(args) >= 2
                and "program_config" not in kw
                and act in (None, "silu")
                and isinstance(args[1], ttnn.Tensor)
                and (
                    args[0].shape[-2] == 1
                    if ROWS == 1
                    else args[0].shape[-2] <= ROWS and math.prod(list(args[0].shape)[:-2]) == 1
                )
            ):
                w = args[1]
                key = (tuple(w.shape), str(w.dtype), str(kw.get("dtype")), act)
                if key not in cache:  # a silu activation goes into the program config (fused, as linear does it)
                    fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act == "silu" else None
                    cache[key] = _config(mesh, w, kw.get("dtype"), fa)
                if cache[key] is not None:
                    kw["program_config"] = cache[key]
                    kw.pop("activation", None)
            return _fn(*args, **kw)

        setattr(ttnn, name, wrapped)
