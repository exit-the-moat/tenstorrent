// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token gated-attention pre-SDPA step, reader (one kernel, four roles by compile-time arg):
//   ROLE 0 (q) / 1 (k) / 2 (v): the head rows of the projection output (row 0 of [1, 1, NH * stride * 32]) packed into
//     the [NH, 256] tile layout SDPA decode / paged_update_cache take (row h = head h, 8 tiles): per head h and column
//     tile j the 32-byte row-0 chunks of faces 0 and 1 of source tile h * stride + j go to row h of packed tile j
//     (NOC copies; the source is L1). The rest of each packed tile comes from MEM_ZEROS. q / k: + the norm weight row
//     (8 tiles), cos / sin tiles 0-1 (the rotary dims) and an all-ones tile for the compute kernel.
//   ROLE 3 (gate): copies the gate tiles of n heads (source tile h * 16 + 8 + j) to the [1, 1, NH * 256] gate row.
//   ROLE 4 (gate, packed): as ROLE 2 for the gate half of q|gate (source tiles h * 16 + 8 + j) -> [16, 256] tiles.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2, cb_ones = 3, cb_out = 16;

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);      // heads packed by this core (q 16, k / v 2)
    constexpr uint32_t STRIDE = get_compile_time_arg_val(2);  // source tiles per head (q|gate: 16, k / v: 8)
    constexpr auto src_a = TensorAccessorArgs<3>();
    constexpr auto w_a = TensorAccessorArgs<src_a.next_compile_time_args_offset()>();
    constexpr auto cos_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto sin_a = TensorAccessorArgs<cos_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<sin_a.next_compile_time_args_offset()>();
    constexpr auto dst_a = TensorAccessorArgs<one_a.next_compile_time_args_offset()>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t one_addr = get_arg_val<uint32_t>(4);
    const uint32_t dst_addr = get_arg_val<uint32_t>(5);
    const uint32_t g0 = get_arg_val<uint32_t>(6);  // ROLE 3: first gate tile of this core
    const uint32_t gn = get_arg_val<uint32_t>(7);  // ROLE 3: gate tiles of this core
    const uint32_t base = get_arg_val<uint32_t>(8);  // first source page (k / v inside a merged q|gate|k|v row)

    const uint32_t tb = get_tile_size(cb_x);
    const auto src = TensorAccessor(src_a, src_addr, tb);
    Noc noc;

    if constexpr (ROLE == 3) {
        const auto dst = TensorAccessor(dst_a, dst_addr, tb);
        CircularBuffer cx(cb_x);
        cx.reserve_back(gn);
        for (uint32_t t = 0; t < gn; t++) {
            const uint32_t g = g0 + t;
            noc.async_read(src, cx, tb, {.page_id = (g / 8) * 16 + 8 + g % 8}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        auto s = use<CircularBuffer::AddrSelector::WRITE_PTR>(cx);
        for (uint32_t t = 0; t < gn; t++) {
            noc.async_write(s, dst, tb, {.offset_bytes = t * tb}, {.page_id = g0 + t});
        }
        noc.async_write_barrier();
        return;
    }

    constexpr uint32_t OFF = ROLE == 4 ? 8 : 0;  // the gate half of each head's q|gate tiles
    CircularBuffer cx(ROLE >= 2 ? cb_out : cb_x);  // v / packed gate need no compute: straight to the writer
    cx.reserve_back(8);
    const uint32_t px = cx.get_write_ptr();
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t j = 0; j < 8; j++) {  // the parts no head row covers: rows NH..15 of faces 0 / 1, faces 2 / 3
        const uint32_t p = px + j * tb;
        if constexpr (NH < 16) {
            noc_async_read(zeros, p + NH * 32, (16 - NH) * 32);
            noc_async_read(zeros, p + 512 + NH * 32, (16 - NH) * 32);
        }
        noc_async_read(zeros, p + 1024, 512);
        noc_async_read(zeros, p + 1536, 512);
    }
    for (uint32_t h = 0; h < NH; h++) {
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t page = base + h * STRIDE + OFF + j;
            noc_async_read(src.get_noc_addr(page, 0), px + j * tb + h * 32, 32);
            noc_async_read(src.get_noc_addr(page, 512), px + j * tb + 512 + h * 32, 32);
        }
    }
    if constexpr (ROLE < 2) {
        const auto w = TensorAccessor(w_a, w_addr, tb);
        const auto cs = TensorAccessor(cos_a, cos_addr, tb);
        const auto sn = TensorAccessor(sin_a, sin_addr, tb);
        const auto one = TensorAccessor(one_a, one_addr, tb);
        CircularBuffer cw(cb_w), ccs(cb_cs), co(cb_ones);
        cw.reserve_back(8);
        ccs.reserve_back(4);
        co.reserve_back(1);
        for (uint32_t j = 0; j < 8; j++) {
            noc.async_read(w, cw, tb, {.page_id = j}, {.offset_bytes = j * tb});
        }
        noc.async_read(cs, ccs, tb, {.page_id = 0}, {.offset_bytes = 0});  // cos tiles 0, 1, sin tiles 0, 1
        noc.async_read(cs, ccs, tb, {.page_id = 1}, {.offset_bytes = tb});
        noc.async_read(sn, ccs, tb, {.page_id = 0}, {.offset_bytes = 2 * tb});
        noc.async_read(sn, ccs, tb, {.page_id = 1}, {.offset_bytes = 3 * tb});
        noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cx.push_back(8);
        cw.push_back(8);
        ccs.push_back(4);
        co.push_back(1);
    } else {
        noc.async_read_barrier();
        cx.push_back(8);
    }
}
