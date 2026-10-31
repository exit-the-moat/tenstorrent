// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: one-token residual add + RMSNorm (h = x + d, y = norm(h) * w) as one program, reader (every core of
// the program does the same; port5's rownorm reader with a second input): the 64 row-0 chunks of 32 values of x
// (and d) are packed into 2 tiles each (row r of packed tile p = chunk 32p + r): 64-byte reads (rows 0-1 of faces 0
// and 1; the source may be DRAM) into aligned staging slots, then 32-byte NOC copies inside L1. Plus the packed norm
// weight (2 tiles, same layout) and an all-ones tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ones = 2, cb_stage = 3, cb_d = 5;

template <typename A>
inline void stage_row(const A& src, uint32_t s0, uint32_t nt) {
    for (uint32_t c = 0; c < nt; c++) {
        noc_async_read(src.get_noc_addr(c, 0), s0 + (2 * c) * 64, 64);
        noc_async_read(src.get_noc_addr(c, 512), s0 + (2 * c + 1) * 64, 64);
    }
}

inline void pack_row(uint32_t s0, uint32_t px, uint32_t nt, uint32_t tb) {
    for (uint32_t c = 0; c < nt; c++) {
        // packed tile c / 32, row r = c % 32: face 0 / 1 (rows 0..15) or face 2 / 3 (rows 16..31)
        const uint32_t r = c % 32, base = px + (c / 32) * tb + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        noc_async_read(get_noc_addr(s0 + (2 * c) * 64), base, 32);
        noc_async_read(get_noc_addr(s0 + (2 * c + 1) * 64), base + 512, 32);
    }
}

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);  // tiles of the row (64)
    constexpr uint32_t HAS_D = get_compile_time_arg_val(1);
    constexpr auto x_a = TensorAccessorArgs<2>();
    constexpr auto d_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t one_addr = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_a, x_addr, tb);
    const auto d = TensorAccessor(d_a, d_addr, tb);
    const auto w = TensorAccessor(w_a, w_addr, tb);
    const auto one = TensorAccessor(one_a, one_addr, tb);
    Noc noc;
    CircularBuffer cx(cb_x), cd(cb_d), cw(cb_w), co(cb_ones), stg(cb_stage);
    cx.reserve_back(NT / 32);
    if constexpr (HAS_D) {
        cd.reserve_back(NT / 32);
    }
    cw.reserve_back(NT / 32);
    co.reserve_back(1);
    stg.reserve_back((HAS_D ? 2 : 1) * NT / 16);  // 2 slots of 64 bytes per chunk and input
    const uint32_t s0 = stg.get_write_ptr(), s1 = s0 + NT * 128;
    stage_row(x, s0, NT);
    if constexpr (HAS_D) {
        stage_row(d, s1, NT);
    }
    for (uint32_t p = 0; p < NT / 32; p++) {
        noc.async_read(w, cw, tb, {.page_id = p}, {.offset_bytes = p * tb});
    }
    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    pack_row(s0, cx.get_write_ptr(), NT, tb);
    if constexpr (HAS_D) {
        pack_row(s1, cd.get_write_ptr(), NT, tb);
    }
    noc.async_read_barrier();
    cx.push_back(NT / 32);
    if constexpr (HAS_D) {
        cd.push_back(NT / 32);
    }
    cw.push_back(NT / 32);
    co.push_back(1);
}
