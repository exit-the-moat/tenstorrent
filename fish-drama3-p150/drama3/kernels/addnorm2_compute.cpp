// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: one-token residual add + RMSNorm split over NC cores (v2), compute of core k on its packed tile:
// H = X + D rounded to bf16 (cb_h, also written out as the new residual); T = ones @ ((H * H) @ ones), every entry
// this core's sum of squares, goes to the writer, which gathers the NC partials into row j of G on every core;
// S = ones @ G (every entry the total), R = rsqrt(S / dim + eps), Y = H * R * W.

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

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ones = 2, cb_d = 5, cb_h = 6, cb_g = 7, cb_out = 16, cb_sq = 24, cb_m = 25,
                   cb_r = 26, cb_t = 27;

inline void matmul_one(uint32_t a, uint32_t b, uint32_t out) {
    tile_regs_acquire();
    matmul_init(a, b, 0);
    matmul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t INV_N = get_compile_time_arg_val(0);
    constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(1);  // (EPS is an LLK macro)
    constexpr uint32_t HAS_D = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = HAS_D ? cb_h : cb_x;
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    CircularBuffer x(cb_x), d(cb_d), hb(cb_h), g(cb_g), w(cb_w), ones(cb_ones), out(cb_out), sq(cb_sq), m(cb_m), r(cb_r),
        t(cb_t);
    x.wait_front(1);
    if constexpr (HAS_D) {
        d.wait_front(1);
        hb.reserve_back(1);
        add_tiles_init(cb_x, cb_d);
        tile_regs_acquire();
        add_tiles(cb_x, cb_d, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_h);
        tile_regs_release();
        hb.push_back(1);
        hb.wait_front(1);  // never popped here: the writer reads it too
    }
    sq.reserve_back(1);
    mul_init(cb_src, cb_src, false);
    tile_regs_acquire();
    mul_tiles(cb_src, cb_src, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sq);
    tile_regs_release();
    sq.push_back(1);
    ones.wait_front(1);
    sq.wait_front(1);
    m.reserve_back(1);
    matmul_one(cb_sq, cb_ones, cb_m);  // row sums, in every column
    m.push_back(1);
    sq.pop_front(1);
    m.wait_front(1);
    t.reserve_back(1);
    matmul_one(cb_ones, cb_m, cb_t);  // this core's total, in every entry
    t.push_back(1);
    m.pop_front(1);
    g.wait_front(1);
    r.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_ones, cb_g, 0);
    matmul_tiles(cb_ones, cb_g, 0, 0, 0);  // the sum of the NC partials, in every entry
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
    g.pop_front(1);
    r.wait_front(1);
    w.wait_front(1);
    out.reserve_back(1);
    tile_regs_acquire();
    mul_init(cb_src, cb_r, false);
    mul_tiles(cb_src, cb_r, 0, 0, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    out.push_back(1);
    x.pop_front(1);
    if constexpr (HAS_D) {
        d.pop_front(1);
    }
    w.pop_front(1);
    ones.pop_front(1);
    r.pop_front(1);
}
