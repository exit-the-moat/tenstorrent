// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: one-token residual add + RMSNorm split over NC cores (v2), reader of core k: only its PER row-0
// chunks c0 .. c0 + PER of x (and d) are packed into one tile each (row i = chunk c0 + i; rows PER..31 zero):
// 64-byte reads (rows 0-1 of faces 0 and 1; the source may be DRAM) into aligned staging slots, then 32-byte NOC
// copies inside L1. Plus this core's tile of the host-packed norm weight (same layout) and an all-ones tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ones = 2, cb_stage = 3, cb_d = 5;

template <typename A>
inline void stage_row(const A& src, uint32_t s0, uint32_t c0, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        noc_async_read(src.get_noc_addr(c0 + i, 0), s0 + (2 * i) * 64, 64);
        noc_async_read(src.get_noc_addr(c0 + i, 512), s0 + (2 * i + 1) * 64, 64);
    }
}

inline void zero_rest(uint32_t p, uint32_t n) {  // rows n..15 of faces 0 / 1, faces 2 / 3
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    noc_async_read(zeros, p + n * 32, (16 - n) * 32);
    noc_async_read(zeros, p + 512 + n * 32, (16 - n) * 32);
    noc_async_read(zeros, p + 1024, 512);
    noc_async_read(zeros, p + 1536, 512);
}

inline void pack_row(uint32_t s0, uint32_t px, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        noc_async_read(get_noc_addr(s0 + (2 * i) * 64), px + i * 32, 32);
        noc_async_read(get_noc_addr(s0 + (2 * i + 1) * 64), px + 512 + i * 32, 32);
    }
}

void kernel_main() {
    constexpr uint32_t PER = get_compile_time_arg_val(0);  // chunks (tiles of the row) per core, <= 16
    constexpr uint32_t HAS_D = get_compile_time_arg_val(1);
    constexpr auto x_a = TensorAccessorArgs<2>();
    constexpr auto d_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t one_addr = get_arg_val<uint32_t>(3);
    const uint32_t k = get_arg_val<uint32_t>(4);
    const uint32_t c0 = k * PER;
    const uint32_t tb = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_a, x_addr, tb);
    const auto d = TensorAccessor(d_a, d_addr, tb);
    const auto w = TensorAccessor(w_a, w_addr, tb);
    const auto one = TensorAccessor(one_a, one_addr, tb);
    Noc noc;
    CircularBuffer cx(cb_x), cd(cb_d), cw(cb_w), co(cb_ones), stg(cb_stage);
    cx.reserve_back(1);
    if constexpr (HAS_D) {
        cd.reserve_back(1);
    }
    cw.reserve_back(1);
    co.reserve_back(1);
    stg.reserve_back(1);  // 2 slots of 64 bytes per chunk and input (<= 4 KB)
    const uint32_t s0 = stg.get_write_ptr(), s1 = s0 + PER * 128;
    stage_row(x, s0, c0, PER);
    if constexpr (HAS_D) {
        stage_row(d, s1, c0, PER);
    }
    zero_rest(cx.get_write_ptr(), PER);
    if constexpr (HAS_D) {
        zero_rest(cd.get_write_ptr(), PER);
    }
    noc.async_read(w, cw, tb, {.page_id = k}, {.offset_bytes = 0});
    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    pack_row(s0, cx.get_write_ptr(), PER);
    if constexpr (HAS_D) {
        pack_row(s1, cd.get_write_ptr(), PER);
    }
    noc.async_read_barrier();
    co.push_back(1);
    cw.push_back(1);
    cx.push_back(1);
    if constexpr (HAS_D) {
        cd.push_back(1);
    }
}
