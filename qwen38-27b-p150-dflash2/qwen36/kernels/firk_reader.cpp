// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_FIR_K=1): prefill FIR conv of the GDN (depthwise causal conv1d, 4 taps) as one program, reader.
// Work unit = 32 output rows (row block r) x W tiles of columns (column block c); units are ordered column-major
// (u = c * NR + r) so a core keeps its 4 x W tap tiles while it walks down the rows. Per unit: 35 rows of x_padded
// (rows 0-2 = the carry, row i >= 3 = x row i - 3; both ROW_MAJOR, one page per row) are staged, W * 64 bytes of each,
// then the 4 shifted 32-row blocks (rows k .. k + 31 for tap k) are one contiguous local copy each into cb_rm + k.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_rm = 0, cb_w = 8, cb_stage = 9, cb_zero = 10;  // cb_rm .. cb_rm + 3: one per tap

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);   // tiles per column block
    constexpr uint32_t NT = get_compile_time_arg_val(1);  // column tiles of the row (D / 32)
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // row bytes (D * 2)
    constexpr auto x_a = TensorAccessorArgs<3>();
    constexpr auto c_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t u1 = get_arg_val<uint32_t>(4);
    const uint32_t NR = get_arg_val<uint32_t>(5);
    constexpr uint32_t tb = 2048, SEG = W * 64;
    const auto x = TensorAccessor(x_a, x_addr, RB);
    const auto cr = TensorAccessor(c_a, c_addr, RB);
    const auto wt = TensorAccessor(w_a, w_addr, tb);
    CircularBuffer cw(cb_w), stg(cb_stage);
    stg.reserve_back(1);
    const uint32_t s0 = stg.get_write_ptr();
    {
        CircularBuffer cz(cb_zero);
        cz.reserve_back(1);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t q = 0; q < tb; q += 512) {
            noc_async_read(zeros, cz.get_write_ptr() + q, 512);
        }
        noc_async_read_barrier();
        cz.push_back(1);
    }
    uint32_t cur = 0xFFFFFFFF;
    for (uint32_t u = u0; u < u1; u++) {
        const uint32_t c = u / NR, r = u % NR;
        if (c != cur) {
            cw.reserve_back(4 * W);  // the CB holds one set: waits until compute popped the previous taps
            const uint32_t p = cw.get_write_ptr();
            for (uint32_t k = 0; k < 4; k++) {
                for (uint32_t j = 0; j < W; j++) {
                    noc_async_read(wt.get_noc_addr(k * NT + c * W + j, 0), p + (k * W + j) * tb, tb);
                }
            }
            noc_async_read_barrier();
            cw.push_back(4 * W);
            cur = c;
        }
        const uint32_t off = c * SEG;
        for (uint32_t i = 0; i < 35; i++) {
            const uint32_t row = 32 * r + i;
            const uint64_t a = row < 3 ? cr.get_noc_addr(row, off) : x.get_noc_addr(row - 3, off);
            noc_async_read(a, s0 + i * SEG, SEG);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer cb(cb_rm + k);
            cb.reserve_back(W);
            noc_async_read(get_noc_addr(s0 + k * SEG), cb.get_write_ptr(), 32 * SEG);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer cb(cb_rm + k);
            cb.push_back(W);
        }
    }
}
