// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: the softmax over the router's top-k scores (row 0 of X: k scores, then -32768; M: the max in all of
// row 0): E = exp(X - M), S = E @ ones (row 0: the sum in every column), V = E * (1 / S). Rows 1..31 are junk.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 2, cb_m = 3, cb_ones = 4, cb_e = 5, cb_s = 6, cb_out = 16;

void kernel_main() {
    compute_kernel_hw_startup(cb_x, cb_m, cb_out);
    CircularBuffer x(cb_x), m(cb_m), ones(cb_ones), e(cb_e), s(cb_s), out(cb_out);
    x.wait_front(1);
    m.wait_front(1);
    e.reserve_back(1);
    tile_regs_acquire();
    sub_init(cb_x, cb_m);
    sub_tiles(cb_x, cb_m, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_e);
    tile_regs_release();
    e.push_back(1);
    x.pop_front(1);
    m.pop_front(1);
    ones.wait_front(1);
    e.wait_front(1);
    s.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_e, cb_ones, 0);
    matmul_tiles(cb_e, cb_ones, 0, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    s.push_back(1);
    ones.pop_front(1);
    s.wait_front(1);
    out.reserve_back(1);
    tile_regs_acquire();
    mul_init(cb_e, cb_s, false);
    mul_tiles(cb_e, cb_s, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    out.push_back(1);
    e.pop_front(1);
    s.pop_front(1);
}
