# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-token RMSNorm on 8 cores (Day 4b; default on, DRAMA3_SHARDED_DECODE_NORM=0 to disable).

An interleaved ttnn.rms_norm of a single [1, 2048] row runs on one core: 36.5 us traced on a P150. Width-sharding
the row over 8 cores, norming, and converting back takes 13.9 us with bit-identical output (tests/bench_norm.py).
The backbone runs 81 such norms per frame, the tower 65.
"""

import os

import ttnn

ENABLED = os.environ.get("DRAMA3_SHARDED_DECODE_NORM", "1") == "1"
# rows of a one-tile decode input that take the sharded path (batch mode sets it to B; default: one row)
ROWS = 1
CORES = 8


class ShardedRowNorm:
    def __init__(self, mesh, dim):
        self.mc = ttnn.create_sharded_memory_config(
            shape=(32, dim // CORES),
            core_grid=ttnn.CoreGrid(y=1, x=CORES),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        bw = dim // CORES // 32
        self.pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(CORES, 1), subblock_w=min(4, bw), block_h=1, block_w=bw, inplace=False
        )

    def sharded(self, x, weight, eps, ckc):
        """The normed row left width-sharded on the 8 cores (port7: input of a DRAM-sharded matmul)."""
        a = ttnn.to_memory_config(x, self.mc)
        y = ttnn.rms_norm(a, epsilon=eps, weight=weight, program_config=self.pc, memory_config=self.mc, compute_kernel_config=ckc)
        ttnn.deallocate(a)
        return y

    def __call__(self, x, weight, eps, ckc, out_mc=None):
        """x: one row (interleaved, any buffer); weight: ROW_MAJOR [1, 1, dim / 32, 32] -> interleaved (out_mc or DRAM)."""
        a = ttnn.to_memory_config(x, self.mc)
        y = ttnn.rms_norm(a, epsilon=eps, weight=weight, program_config=self.pc, memory_config=self.mc, compute_kernel_config=ckc)
        ttnn.deallocate(a)
        out = ttnn.to_memory_config(y, out_mc or ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out


class DecodeNormWrapper:
    """Wraps a models.common RMSNorm: one-token calls go through ShardedRowNorm, everything else unchanged."""

    def __init__(self, norm, row_norm):
        self.norm, self.row = norm, row_norm

    def __call__(self, x, mode, in_sharded=False, out_sharded=False, norm_config=None):
        if x.shape[-2] <= ROWS and not in_sharded and not out_sharded:
            out_mc = norm_config.get("output_mem_config") if norm_config else None
            return self.row(x, self.norm.weight, self.norm.eps, self.norm.compute_kernel_config_hifi2, out_mc)
        return self.norm(x, mode=mode, in_sharded=in_sharded, out_sharded=out_sharded, norm_config=norm_config)

    def __getattr__(self, name):
        return getattr(self.norm, name)
