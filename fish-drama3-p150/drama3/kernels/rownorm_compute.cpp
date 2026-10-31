// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token RMSNorm, compute on the 2 packed tiles X (all 2048 values): S = sum of X * X as
// ones @ (sum_p (X_p * X_p) @ ones) (every entry the total), R = rsqrt(S / 2048 + eps), Y_p = X_p * R * W_p.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ones = 2, cb_out = 16, cb_sq = 24, cb_m = 25, cb_r = 26;

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);  // packed tiles (2)
    constexpr uint32_t INV_N = get_compile_time_arg_val(1);
    constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(2);  // (EPS is an LLK macro)
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    CircularBuffer x(cb_x), w(cb_w), ones(cb_ones), out(cb_out), sq(cb_sq), m(cb_m), r(cb_r);
    x.wait_front(NP);
    w.wait_front(NP);
    ones.wait_front(1);
    sq.reserve_back(NP);
    mul_init(cb_x, cb_x, false);
    for (uint32_t p = 0; p < NP; p++) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, p, p, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    sq.push_back(NP);
    sq.wait_front(NP);
    m.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_sq, cb_ones, 0);
    for (uint32_t p = 0; p < NP; p++) {
        matmul_tiles(cb_sq, cb_ones, p, 0, 0);  // row sums, in every column
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_m);
    tile_regs_release();
    m.push_back(1);
    sq.pop_front(NP);
    m.wait_front(1);
    r.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_ones, cb_m, 0);
    matmul_tiles(cb_ones, cb_m, 0, 0, 0);  // column sums of the row sums: the total, in every entry
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_N);
    add_unary_tile(0, NORM_EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    r.push_back(1);
    m.pop_front(1);
    r.wait_front(1);
    out.reserve_back(NP);
    for (uint32_t p = 0; p < NP; p++) {
        tile_regs_acquire();
        mul_init(cb_x, cb_r, false);
        mul_tiles(cb_x, cb_r, p, 0, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, p, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    out.push_back(NP);
    x.pop_front(NP);
    w.pop_front(NP);
    ones.pop_front(1);
    r.pop_front(1);
}
