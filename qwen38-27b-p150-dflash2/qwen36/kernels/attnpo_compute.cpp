// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused one-token gated-attention post-SDPA step (port of drama3 attn_post_compute.cpp), compute:
// P = A * sigmoid(G) on one packed tile. sigmoid(G) is packed (rounded to bf16) before the multiply, as the op chain
// does.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_a = 0, cb_g = 1, cb_s = 3, cb_out = 16;

void kernel_main() {
    compute_kernel_hw_startup(cb_g, cb_a, cb_out);
    CircularBuffer ca(cb_a), cg(cb_g), cs(cb_s), co(cb_out);
    cg.wait_front(1);
    cs.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_g);
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
    mul_tiles_init(cb_a, cb_s);
    tile_regs_acquire();
    mul_tiles(cb_a, cb_s, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    co.push_back(1);
    ca.pop_front(1);
    cs.pop_front(1);
}
