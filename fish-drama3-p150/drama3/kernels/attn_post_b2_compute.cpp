// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: attn_post_compute.cpp looped over the n rows of this core (runtime arg 0; attn_post_b2_*).
// drama3 port5: fused one-token gated-attention post-SDPA step, compute: P = A * sigmoid(G) on one packed tile.
// sigmoid(G) is packed (rounded to bf16) before the multiply, as the op chain does: moving the fp32 DST value into a
// source register instead (dest reuse) truncates it, and the tower heads measurably disagreed more with the host
// tower that way (tests.check_tower_device: top-1 150 / 180 vs 163).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_a = 0, cb_g = 1, cb_s = 3, cb_out = 16;

void kernel_main() {
    compute_kernel_hw_startup(cb_g, cb_a, cb_out);
    CircularBuffer ca(cb_a), cg(cb_g), cs(cb_s), co(cb_out);
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t it = 0; it < n; it++) {
    cg.wait_front(1);
    cs.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_g);
    copy_tile(cb_g, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    cs.push_back(1);
    cg.pop_front(1);
    ca.wait_front(1);
    cs.wait_front(1);
    co.reserve_back(1);
    tile_regs_acquire();
    mul_init(cb_a, cb_s, false);
    mul_tiles(cb_a, cb_s, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    co.push_back(1);
    ca.pop_front(1);
    cs.pop_front(1);
    }
}
