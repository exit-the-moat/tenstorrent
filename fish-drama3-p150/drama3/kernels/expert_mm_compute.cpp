// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token indexed expert matmul, compute: per job, the K-tile products accumulate in one DST tile
// (fp32), weight blocks of KB tiles as the reader streams them.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_a = 0, cb_w = 1, cb_w2 = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t SHARED_A = get_compile_time_arg_val(2);
    const uint32_t nj = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);  // as ttnn's bmm.cpp: in1 -> srcA, in0 -> srcB
    CircularBuffer ca(cb_a), cw(cb_w), cw2(cb_w2), co(cb_out);
    matmul_init(cb_a, cb_w);
    if constexpr (SHARED_A) {
        ca.wait_front(KT);
    }
    for (uint32_t q = 0; q < nj; q++) {
        if constexpr (!SHARED_A) {
            ca.wait_front(KT);
        }
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < KT; k0 += KB) {
            cw.wait_front(KB / 2);
            cw2.wait_front(KB / 2);
#ifndef EMM_NO_MATH
            for (uint32_t k = 0; k < KB / 2; k++) {  // even K tiles from the reader, odd ones from the writer
                matmul_tiles(cb_a, cb_w, k0 + 2 * k, k, 0);
                matmul_tiles(cb_a, cb_w2, k0 + 2 * k + 1, k, 0);
            }
#endif
            cw.pop_front(KB / 2);
            cw2.pop_front(KB / 2);
        }
        tile_regs_commit();
        co.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        co.push_back(1);
        if constexpr (!SHARED_A) {
            ca.pop_front(KT);
        }
    }
    if constexpr (SHARED_A) {
        ca.pop_front(KT);
    }
}
