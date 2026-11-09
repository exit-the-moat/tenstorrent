// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 (Exit the Moat, Episode 2): fused one-token gated-attention pre-SDPA step (QWEN36_ATTN_FUSED=1), reader. Port of
// the Episode 0 Drama 3 kernel (drama3/kernels/attn_pre_reader.cpp) to up to 32 heads (rows 16..31 in faces 2 / 3).
// One kernel, roles by compile-time arg:
//   ROLE 0 (q) / 1 (k) / 2 (v): the head rows of the projection output (row 0 of [1, 1, NH * STRIDE * 32]) packed into
//     the [NH, 256] tile layout SDPA decode / paged_update_cache take (row h = head h, 8 tiles): per head h and column
//     tile j the 32-byte row-0 chunks of faces 0 and 1 of source tile h * STRIDE + j go to row h of packed tile j (NOC
//     copies; the source is L1). The rest of each packed tile comes from MEM_ZEROS. q / k: + the norm weight row
//     (8 tiles), cos / sin tiles 0-1 (the rotary dims) and an all-ones tile for the compute kernel.
//   ROLE 4 (gate, packed): as ROLE 2 for the gate (source tiles h * STRIDE + GOFF + j).
// Batch (the TP path): the source row SROW (= sequence b) instead of row 0, cos / sin tiles CS0, CS0 + 1 (row 0).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2, cb_ones = 3, cb_stage = 5, cb_out = 16;

// byte offset of row r, column half c (0: columns 0..15, 1: 16..31) in a bf16 tile
constexpr uint32_t rowoff(uint32_t r, uint32_t c) { return (r >> 4) * 1024 + c * 512 + (r & 15) * 32; }

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);      // heads packed by this core (<= 32)
    constexpr uint32_t STRIDE = get_compile_time_arg_val(2);  // source tiles per head (q|gate: 16, k / v: 8)
    constexpr uint32_t GOFF = get_compile_time_arg_val(3);    // ROLE 4: first gate tile inside a head's tiles
    constexpr auto src_a = TensorAccessorArgs<4>();
    constexpr auto w_a = TensorAccessorArgs<src_a.next_compile_time_args_offset()>();
    constexpr auto cos_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto sin_a = TensorAccessorArgs<cos_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<sin_a.next_compile_time_args_offset()>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t one_addr = get_arg_val<uint32_t>(4);
    const uint32_t base = get_arg_val<uint32_t>(5);  // first source page
    const uint32_t srow = get_arg_val<uint32_t>(6);  // source row (sequence)
    const uint32_t cs0 = get_arg_val<uint32_t>(7);   // first cos / sin page of this sequence

    constexpr uint32_t tb = 2048;
    const auto src = TensorAccessor(src_a, src_addr, tb);
    Noc noc;

    constexpr uint32_t OFF = ROLE == 4 ? GOFF : 0;
    CircularBuffer cx(ROLE >= 2 ? cb_out : cb_x);  // v / packed gate need no compute: straight to the writer
    cx.reserve_back(8);
    const uint32_t px = cx.get_write_ptr();
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t j = 0; j < 8; j++) {  // the rows no head covers: NH..31
        const uint32_t p = px + j * tb;
        if constexpr (NH < 16) {
            noc_async_read(zeros, p + NH * 32, (16 - NH) * 32);
            noc_async_read(zeros, p + 512 + NH * 32, (16 - NH) * 32);
        }
        constexpr uint32_t r2 = NH > 16 ? NH - 16 : 0;
        if constexpr (r2 < 16) {
            noc_async_read(zeros, p + 1024 + r2 * 32, (16 - r2) * 32);
            noc_async_read(zeros, p + 1536 + r2 * 32, (16 - r2) * 32);
        }
    }
    // the source may be DRAM: 32-byte DRAM reads into 32-byte (not 64-byte) aligned L1 slots corrupt the data, so
    // read the 64-byte row pair containing srow into aligned staging slots, then copy the 32-byte row inside L1
    CircularBuffer stg(cb_stage);
    stg.reserve_back(12);  // 2 x 64 bytes per (head, tile): NH * 8 * 128 <= 24 KB
    const uint32_t sb = stg.get_write_ptr();
    const uint32_t pr = (srow & ~1u), sr = (srow & 1u) * 32;
    for (uint32_t h = 0; h < NH; h++) {
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t page = base + h * STRIDE + OFF + j, sl = sb + (h * 8 + j) * 128;
            noc_async_read(src.get_noc_addr(page, rowoff(pr, 0)), sl, 64);
            noc_async_read(src.get_noc_addr(page, rowoff(pr, 1)), sl + 64, 64);
        }
    }
    noc.async_read_barrier();
    for (uint32_t h = 0; h < NH; h++) {
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t sl = sb + (h * 8 + j) * 128;
            noc_async_read(get_noc_addr(sl + sr), px + j * tb + rowoff(h, 0), 32);
            noc_async_read(get_noc_addr(sl + 64 + sr), px + j * tb + rowoff(h, 1), 32);
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
        noc.async_read(cs, ccs, tb, {.page_id = cs0}, {.offset_bytes = 0});  // cos tiles 0, 1, sin tiles 0, 1
        noc.async_read(cs, ccs, tb, {.page_id = cs0 + 1}, {.offset_bytes = tb});
        noc.async_read(sn, ccs, tb, {.page_id = cs0}, {.offset_bytes = 2 * tb});
        noc.async_read(sn, ccs, tb, {.page_id = cs0 + 1}, {.offset_bytes = 3 * tb});
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
