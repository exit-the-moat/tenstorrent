// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10 bb12 (DRAMA3_SAMPLER_FUSED=1): E = exp(X - M) on one fp32 tile (X: the 1024 logits, M: their max in
// every element), the SFPU sub + exp that ttnn.subtract / ttnn.exp run on fp32 (unpacked to DST as fp32, fp32 DST);
// then P = V / Z and S = V / N (div_binary_tile, as ttnn.divide; V: the top-32 e in row 0, Z: z, N: the noise row).
// The reader (kernels/sampler_fused_reader.cpp) does the rest.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_m = 1, cb_e = 2, cb_v = 6, cb_z = 7, cb_n = 8, cb_p = 9, cb_s = 10;

void kernel_main() {
    compute_kernel_hw_startup(cb_x, cb_m, cb_e);
    CircularBuffer x(cb_x), m(cb_m), e(cb_e);
    x.wait_front(1);
    m.wait_front(1);
    e.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_x);
    copy_tile(cb_x, 0, 0);
    copy_tile_to_dst_init_short(cb_m);
    copy_tile(cb_m, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_e);
    tile_regs_release();
    e.push_back(1);
    x.pop_front(1);
    m.pop_front(1);

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
