// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused gated-attention post-SDPA step for B sequences (TP path), compute: P_b = A_b * sigmoid(G_b) per packed
// tile; sigmoid(G) is packed (rounded to bf16) before the multiply, as the op chain does.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_a = 0, cb_g = 1, cb_s = 3, cb_out = 16;

void kernel_main() {
    const uint32_t B = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_g, cb_a, cb_out);
    CircularBuffer ca(cb_a), cg(cb_g), cs(cb_s), co(cb_out);
    cg.wait_front(B);
    cs.reserve_back(B);
    copy_tile_to_dst_init_short(cb_g);
    sigmoid_tile_init();
    for (uint32_t b = 0; b < B; b++) {
        tile_regs_acquire();
        copy_tile(cb_g, b, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s);
        tile_regs_release();
    }
    cs.push_back(B);
    cg.pop_front(B);
    ca.wait_front(B);
    cs.wait_front(B);
    co.reserve_back(B);
    mul_tiles_init(cb_a, cb_s);
    for (uint32_t b = 0; b < B; b++) {
        tile_regs_acquire();
        mul_tiles(cb_a, cb_s, b, b, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    co.push_back(B);
    ca.pop_front(B);
    cs.pop_front(B);
}
