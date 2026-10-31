// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10 bb12 (DRAMA3_SAMPLER_FUSED_CORES): the compute side of kernels/sampler_mc_reader.cpp. Every core:
// E_j = exp(X_j - M_j). The central core (runtime arg 0 == 0) also: V = exp(G - MG) (the top-32 e and the g_j), then
// P = V / Z and S = V / N (div_binary_tile). SFPU ops on fp32 tiles unpacked to DST as fp32, as ttnn runs them.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_m = 1, cb_e = 2, cb_v = 6, cb_z = 7, cb_n = 8, cb_p = 9, cb_s = 10, cb_g = 11, cb_mg = 12;

static void exp_diff(uint32_t ca, uint32_t cb, uint32_t co) {
    CircularBuffer a(ca), b(cb), o(co);
    a.wait_front(1);
    b.wait_front(1);
    o.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(ca);
    copy_tile(ca, 0, 0);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, co);
    tile_regs_release();
    o.push_back(1);
    a.pop_front(1);
    b.pop_front(1);
}

void kernel_main() {
    const uint32_t j = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_m, cb_e);
    exp_diff(cb_x, cb_m, cb_e);
    if (j != 0) {
        return;
    }
    exp_diff(cb_g, cb_mg, cb_v);
    CircularBuffer v(cb_v), z(cb_z), n(cb_n), p(cb_p), sc(cb_s);
    v.wait_front(1);
    z.wait_front(1);
    n.wait_front(1);
    p.reserve_back(1);
    sc.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_v);
    copy_tile(cb_v, 0, 0);
    copy_tile(cb_v, 0, 2);
    copy_tile_to_dst_init_short(cb_z);
    copy_tile(cb_z, 0, 1);
    copy_tile_to_dst_init_short(cb_n);
    copy_tile(cb_n, 0, 3);
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
    div_binary_tile(2, 3, 2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_p);
    pack_tile(2, cb_s);
    tile_regs_release();
    p.push_back(1);
    sc.push_back(1);
    v.pop_front(1);
    z.pop_front(1);
    n.pop_front(1);
}
