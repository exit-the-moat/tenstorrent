// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token MoE combine, compute. Per hidden tile: DST0 = shared * sigmoid(g), then
// += down_j * v_j for the k slots: FPU scalar-broadcast multiplies by [0,0] of the scalar tiles, which on
// Blackhole accumulate into the fp32 DST tile (so the sum needs no add). Replaces slice + mul + fill-pad + sum over the slots + sigmoid + mul + add (~26 us per layer).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_g = 0, cb_v = 1, cb_d = 2, cb_sh = 3, cb_gs = 5, cb_out = 16;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);  // hidden tiles per core

    compute_kernel_hw_startup(cb_d, cb_v, cb_out);
    CircularBuffer g(cb_g), v(cb_v), d(cb_d), sh(cb_sh), gs(cb_gs), out(cb_out);
    // sigmoid(g) once (at [0,0])
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
    v.wait_front(K);
    for (uint32_t i = 0; i < n; i++) {
        d.wait_front(K);
        sh.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        mul_bcast_scalar_init(cb_sh, cb_gs);
        mul_tiles_bcast_scalar(cb_sh, cb_gs, 0, 0, 0);
        for (uint32_t j = 0; j < K; j++) {
            mul_bcast_scalar_init(cb_d, cb_v);
            mul_tiles_bcast_scalar(cb_d, cb_v, j, j, 0);  // the FPU broadcast multiply accumulates into DST (BH)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        d.pop_front(K);
        sh.pop_front(1);
    }
    gs.pop_front(1);
    v.pop_front(K);
}
