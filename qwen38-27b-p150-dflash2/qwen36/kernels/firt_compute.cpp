// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_FIR_T=1): firk_compute.cpp with the x row blocks untilized here first (tiles of qkvzab from the
// reader -> cb_u, row-major, for the reader's shifted-block assembly; one extra block at the start of a core's
// range). Then as QWEN36_FIR_K=1: GDN prefill FIR conv, compute. Per unit: tilize the 4 shifted row-major blocks (W tiles
// each), then per tile the same math as the stock op chain, in the same order and dest format (16-bit dest, as the
// bf16 stock ops run): out = x0 * w0 (SFPU binary mul, as ttnn.multiply), out = out + x_k * w_k for k = 1..3 (SFPU addcmul,
// value 1.0, round to nearest bf16, as ttnn.addcmul), silu (as ttnn.silu). w_k tiles are row-replicated on the host.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_rm = 0, cb_t = 4, cb_w = 8, cb_tin = 10, cb_u = 11, cb_out = 16;  // cb_rm + k, cb_t + k: tap k

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t ONE = get_compile_time_arg_val(1);  // bits of 1.0f (the addcmul value)
    constexpr uint32_t DBG = get_compile_time_arg_val(2);  // debug: bit 0 skips the addcmul taps, bit 1 skips silu
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t NR = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(cb_rm, cb_w, cb_out);
    CircularBuffer cw(cb_w), co(cb_out);
    uint32_t cur = 0xFFFFFFFF;
    CircularBuffer cti(cb_tin), cu(cb_u);
    auto until = [&]() {
        cti.wait_front(W);
        cu.reserve_back(W);
        untilize_init(cb_tin);
        untilize_block(cb_tin, W, cb_u);
        untilize_uninit(cb_tin);
        cu.push_back(W);
        cti.pop_front(W);
    };
    for (uint32_t u = u0; u < u1; u++) {
        const uint32_t c = u / NR;
        if ((u % NR) > 0 && u == u0) {
            until();
        }
        until();
        if (c != cur) {
            if (cur != 0xFFFFFFFF) {
                cw.pop_front(4 * W);
            }
            cw.wait_front(4 * W);
            cur = c;
        }
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer ci(cb_rm + k), ct(cb_t + k);
            ci.wait_front(W);
            ct.reserve_back(W);
            if constexpr (!(DBG & 4)) {  // debug (timing only): bit 2 skips the shifted-block tilizes
                tilize_init(cb_rm + k, W, cb_t + k);
                tilize_block(cb_rm + k, W, cb_t + k);
                tilize_uninit(cb_rm + k, cb_t + k);
            }
            ct.push_back(W);
            ci.pop_front(W);
        }
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer ct(cb_t + k);
            ct.wait_front(W);
        }
        co.reserve_back(W);
        for (uint32_t j = 0; j < W; j++) {
            tile_regs_acquire();
            // tap 0: x0 * w0 as SFPU binary mul (ttnn.multiply of bf16 runs on the SFPU unless fast_and_approximate)
            copy_tile_to_dst_init_short(cb_t);
            copy_tile(cb_t, j, 0);
            copy_tile_to_dst_init_short(cb_w);
            copy_tile(cb_w, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            for (uint32_t k = 1; k < ((DBG & 1) ? 1 : 4); k++) {
                copy_tile_to_dst_init_short(cb_t + k);
                copy_tile(cb_t + k, j, 1);
                copy_tile_to_dst_init_short(cb_w);
                copy_tile(cb_w, k * W + j, 2);
                addcmul_tile_init();
                addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, ONE);
            }
            if constexpr (!(DBG & 2)) {
                silu_tile_init();
                silu_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        co.push_back(W);
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer ct(cb_t + k);
            ct.pop_front(W);
        }
    }
}
