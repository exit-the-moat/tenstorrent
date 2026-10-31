// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token RMSNorm of a [1, 1, 2048] row, reader (every core of the program does the same): the 64
// row-0 chunks of 32 values are packed into 2 tiles (row r of packed tile p = chunk 32p + r): 64-byte reads (rows 0-1
// of faces 0 and 1; the source may be DRAM) into aligned staging slots, then 32-byte NOC copies inside L1. Plus the
// packed norm weight (2 tiles, same layout) and an all-ones tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ones = 2, cb_stage = 3;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);  // tiles of the row (64)
    constexpr auto x_a = TensorAccessorArgs<1>();
    constexpr auto w_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t one_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_a, x_addr, tb);
    const auto w = TensorAccessor(w_a, w_addr, tb);
    const auto one = TensorAccessor(one_a, one_addr, tb);
    Noc noc;
    CircularBuffer cx(cb_x), cw(cb_w), co(cb_ones), stg(cb_stage);
    cx.reserve_back(NT / 32);
    cw.reserve_back(NT / 32);
    co.reserve_back(1);
    stg.reserve_back(NT / 16);  // 2 slots of 64 bytes per chunk
    const uint32_t s0 = stg.get_write_ptr(), px = cx.get_write_ptr();
    for (uint32_t c = 0; c < NT; c++) {
        noc_async_read(x.get_noc_addr(c, 0), s0 + (2 * c) * 64, 64);
        noc_async_read(x.get_noc_addr(c, 512), s0 + (2 * c + 1) * 64, 64);
    }
    for (uint32_t p = 0; p < NT / 32; p++) {
        noc.async_read(w, cw, tb, {.page_id = p}, {.offset_bytes = p * tb});
    }
    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    for (uint32_t c = 0; c < NT; c++) {
        // packed tile c / 32, row r = c % 32: face 0 / 1 (rows 0..15) or face 2 / 3 (rows 16..31)
        const uint32_t r = c % 32, base = px + (c / 32) * tb + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        noc_async_read(get_noc_addr(s0 + (2 * c) * 64), base, 32);
        noc_async_read(get_noc_addr(s0 + (2 * c + 1) * 64), base + 512, 32);
    }
    noc.async_read_barrier();
    cx.push_back(NT / 32);
    cw.push_back(NT / 32);
    co.push_back(1);
}
