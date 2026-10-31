// drama3 port10: moe_combine_s = moe_combine with the shared expert's rows at page SOFF + c of the shared tensor
// (DRAMA3_MOE_SHARED_IN_EMM=1: slot k of the ExpertMM down output).
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token MoE combine, reader. out = sum_j v_j * down_j + sigmoid(g) * shared over the hidden tiles
// [t0, t0 + n) of this core. Pushes the shared-gate tile (g at [0,0]), the k routing weights as k scalar tiles
// (v_j at [0,0], cut out of row 0 of the softmax tile), then per hidden tile c: the k compact down rows (slot j's
// tile c of [1, k, 32, H]) and the shared expert's tile c.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_g = 0, cb_v = 1, cb_d = 2, cb_sh = 3, cb_scr = 4;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);  // hidden tiles
    constexpr uint32_t SOFF = K * HT;
    constexpr auto d_a = TensorAccessorArgs<2>();
    constexpr auto v_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto sh_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<sh_a.next_compile_time_args_offset()>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t sh_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);

    const uint32_t tb = get_tile_size(cb_d);
    const auto d = TensorAccessor(d_a, d_addr, tb);
    const auto v = TensorAccessor(v_a, v_addr, tb);
    const auto sh = TensorAccessor(sh_a, sh_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);

    Noc noc;
    CircularBuffer cg(cb_g), cv(cb_v), cd(cb_d), csh(cb_sh), scr(cb_scr);
    cg.reserve_back(1);
    scr.reserve_back(1);
    noc.async_read(g, cg, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(v, scr, tb, {.page_id = 0}, {.offset_bytes = 0});
    cv.reserve_back(K);
    // scalar tiles: only [0,0] is read (FPU scalar broadcast), the rest of each tile is left as it is
    noc.async_read_barrier();
    cg.push_back(1);
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cv.get_write_ptr());
    for (uint32_t j = 0; j < K; j++) {
        dst[j * (tb / 2)] = src[j];  // K <= 16: row 0, face 0
    }
    cv.push_back(K);

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = t0 + i;
        cd.reserve_back(K);
        csh.reserve_back(1);
        for (uint32_t j = 0; j < K; j++) {
            noc.async_read(d, cd, tb, {.page_id = j * HT + c}, {.offset_bytes = j * tb});
        }
        noc.async_read(sh, csh, tb, {.page_id = SOFF + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cd.push_back(K);
        csh.push_back(1);
    }
}
