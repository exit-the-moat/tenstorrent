// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: MoE combine for B decode rows (kernels/moe_combine_b_reader.cpp), compute. Per hidden tile:
// DST0 = shared * colbcast(sigmoid(g)) + sum_s down_s * colbcast(w_s), w_s = v[b, j] at (b, 0) (zero elsewhere).
// MODE 0: the FPU column-broadcast multiplies accumulate into DST0 (as the scalar-broadcast ones do on Blackhole,
// kernels/moe_combine_compute.cpp); MODE 1: each product into DST1 (zero-filled first), added on the SFPU in fp32.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_g = 0, cb_w = 1, cb_d = 2, cb_sh = 3, cb_gs = 5, cb_out = 16;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);  // B * K slots
    constexpr uint32_t n = get_compile_time_arg_val(1);  // hidden tiles per core
    constexpr uint32_t MODE = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_d, cb_w, cb_out);
    CircularBuffer g(cb_g), w(cb_w), d(cb_d), sh(cb_sh), gs(cb_gs), out(cb_out);
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
        for (uint32_t s = 0; s < S; s++) {
            if constexpr (MODE == 0) {
                mul_bcast_cols_init(cb_d, cb_w);
                mul_tiles_bcast_cols(cb_d, cb_w, s, s, 0);
            } else {
                fill_tile_init();
                fill_tile(1, 0.0f);
                mul_bcast_cols_init(cb_d, cb_w);
                mul_tiles_bcast_cols(cb_d, cb_w, s, s, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
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
