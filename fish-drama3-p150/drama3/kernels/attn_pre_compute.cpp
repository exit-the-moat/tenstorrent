// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token gated-attention pre-SDPA step, compute (q or k core): on the packed [heads, 256]
// tiles X (8 tiles, row h = head h):
//   rms norm per row: R = rsqrt(rowsum(X * X) / 256 + eps) (row sums as (X * X) @ ones, so every column carries
//   its row's sum), N = X * R * w (w: the norm weight row, broadcast over the rows);
//   partial rotary (rotary dims 0..63 = tiles 0, 1; the half rotation moves whole tiles):
//   E0 = N0 cos0 - N1 sin0, E1 = N1 cos1 + N0 sin1, E_j = N_j for j >= 2 (cos = 1, sin = 0 there).
// Same values as the op chain's x * C + (x @ P) * S with the +-1 permutation P.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2, cb_ones = 3, cb_out = 16, cb_sq = 24, cb_r = 25, cb_t = 26, cb_n = 27;

void kernel_main() {
    constexpr uint32_t INV_D = get_compile_time_arg_val(0);  // 1 / 256 (fp32 bits)
    constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(1);  // (EPS is an LLK macro)

    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    CircularBuffer x(cb_x), w(cb_w), cs(cb_cs), ones(cb_ones), out(cb_out), sq(cb_sq), r(cb_r), t(cb_t), n(cb_n);
    x.wait_front(8);
    w.wait_front(8);
    cs.wait_front(4);
    ones.wait_front(1);

    // X * X
    sq.reserve_back(8);
    mul_init(cb_x, cb_x, false);
    for (uint32_t j = 0; j < 8; j++) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    sq.push_back(8);
    sq.wait_front(8);
    // R = rsqrt(rowsum / 256 + eps)
    r.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_sq, cb_ones, 0);
    for (uint32_t j = 0; j < 8; j++) {
        matmul_tiles(cb_sq, cb_ones, j, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_D);
    add_unary_tile(0, NORM_EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    r.push_back(1);
    sq.pop_front(8);
    r.wait_front(1);
    // T = X * R, N = T * w (rows)
    t.reserve_back(8);
    mul_init(cb_x, cb_r, false);
    for (uint32_t j = 0; j < 8; j++) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_r, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_t);
        tile_regs_release();
    }
    t.push_back(8);
    t.wait_front(8);
    n.reserve_back(8);
    mul_bcast_rows_init(cb_t, cb_w);
    for (uint32_t j = 0; j < 8; j++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_t, cb_w, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_n);
        tile_regs_release();
    }
    n.push_back(8);
    t.pop_front(8);
    x.pop_front(8);
    r.pop_front(1);
    n.wait_front(8);

    // rotary: E0 = N0 cos0 - N1 sin0, E1 = N1 cos1 + N0 sin1 (cs: cos0, cos1, sin0, sin1)
    out.reserve_back(8);
    for (uint32_t k = 0; k < 2; k++) {
        tile_regs_acquire();
        mul_bcast_rows_init(cb_n, cb_cs);
        mul_tiles_bcast_rows(cb_n, cb_cs, k, k, 0);
        mul_tiles_bcast_rows(cb_n, cb_cs, 1 - k, 2 + k, 1);
        if (k == 0) {
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    copy_init(cb_n);
    for (uint32_t j = 2; j < 8; j++) {
        tile_regs_acquire();
        copy_tile(cb_n, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    out.push_back(8);
    n.pop_front(8);
    w.pop_front(8);
    cs.pop_front(4);
    ones.pop_front(1);
}
