// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: MoE combine over the deduplicated expert slots (kernels/moe_combine_b3_reader.cpp), compute: as
// moe_combine_b_compute.cpp MODE 0, but over the U slots the reader found (cb 6) instead of all B * K. Per hidden
// tile: DST0 = shared * colbcast(sigmoid(g)) + sum_u down_u * colbcast(w_u). The CBs move S = B * K tiles per push
// (only the first U are used).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_g = 0, cb_w = 1, cb_d = 2, cb_sh = 3, cb_gs = 5, cb_cnt = 6, cb_out = 16;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);  // B * K
    constexpr uint32_t n = get_compile_time_arg_val(1);  // hidden tiles per core

    compute_kernel_hw_startup(cb_d, cb_w, cb_out);
    CircularBuffer g(cb_g), w(cb_w), d(cb_d), sh(cb_sh), gs(cb_gs), cc(cb_cnt), out(cb_out);
    cc.wait_front(1);
    const uint32_t U = read_tile_value(cb_cnt, 0, 0);
    cc.pop_front(1);
    g.wait_front(1);
    gs.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_g);
    copy_tile(cb_g, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_gs);
    tile_regs_release();
    gs.push_back(1);
    g.pop_front(1);
    gs.wait_front(1);
    w.wait_front(S);
    for (uint32_t i = 0; i < n; i++) {
        d.wait_front(S);
        sh.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        mul_bcast_cols_init(cb_sh, cb_gs);
        mul_tiles_bcast_cols(cb_sh, cb_gs, 0, 0, 0);
        for (uint32_t u = 0; u < U; u++) {
            mul_bcast_cols_init(cb_d, cb_w);
            mul_tiles_bcast_cols(cb_d, cb_w, u, u, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        d.pop_front(S);
        sh.pop_front(1);
    }
    gs.pop_front(1);
    w.pop_front(S);
}
