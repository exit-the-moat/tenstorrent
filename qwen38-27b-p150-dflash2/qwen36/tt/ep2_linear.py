# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified for Exit the Moat Episode 2 (2026-11-01): new file.
"""ep2 (QWEN36_SD_MM_CFG=1): explicit 1D decode program configs for the single-device skinny (M <= 32) matmuls.

The single-device code (mlp.py, the experimental GDN / gated-attention modules) calls ttnn.linear without a program
config, so ttnn picks one. For the row-parallel shapes that choice runs at ~270 GB/s; a 1D mcast config on 33 cores
runs at ~370-390 GB/s with bit-identical output (mmbench.py, card 0: down-proj 347 -> 241 us, GDN/attention out-proj
125 -> 90 us). install(module) swaps the module's `ttnn` for a proxy whose linear() adds that config; every other
attribute is the real ttnn.
"""
import ttnn

# (K, N) -> cores, measured (fastest measured, bit-identical to auto)
TABLE = {(17408, 5120): 33, (6144, 5120): 33}
_PC = {}


def _pc(x, w):
    if not hasattr(w, "shape") or len(w.shape) < 2:
        return None
    K, N = int(w.shape[-2]), int(w.shape[-1])
    nc = TABLE.get((K, N))
    if nc is None or x.is_sharded() or int(x.padded_shape[-2]) > ttnn.TILE_SIZE:
        return None
    key = (K, N, nc)
    if key not in _PC:
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        gw = x.device().compute_with_storage_grid_size().x
        _PC[key] = tpc.create_matmul_1d_decode_progcfg(ttnn.TILE_SIZE, K, N, num_cores=nc, grid_w=gw)
    return _PC[key]


def linear(x, w, *args, **kw):
    if kw.get("program_config") is None and "activation" not in kw and not args:
        pc = _pc(x, w)
        if pc is not None:
            kw["program_config"] = pc
    return ttnn.linear(x, w, *args, **kw)


class _Proxy:
    def __getattr__(self, name):
        return getattr(ttnn, name)

    linear = staticmethod(linear)


PROXY = _Proxy()


def install(*modules):
    for m in modules:
        m.ttnn = PROXY
