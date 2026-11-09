// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_FIR_T=1): GDN prefill FIR conv on the TILE projection output (full chunks), reader. Like
// firk_reader.cpp, but x comes as tiles straight from qkvzab (column tiles c * W .. of row block r, row stride NQ): the
// compute kernel untilizes each row block into cb_u, and this kernel assembles the 4 shifted 32-row blocks of x_padded
// from the previous block's last 3 rows (kept in a scratch area; the carry rows for r = 0) and the current block, two
// contiguous local copies each. Units walk down the rows of a column block (u = c * NR + r), so the previous block is
// usually the one just done; at the start of a core's range (r > 0) the compute kernel untilizes block r - 1 once more.
// The last row block's last 3 rows are written as the new carry state (ROW_MAJOR [1, 3, D]).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_rm = 0, cb_w = 8, cb_prev = 9, cb_tin = 10, cb_u = 11;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);  // column tiles of qkv (D / 32)
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // carry / state row bytes (D * 2)
    constexpr uint32_t NQ = get_compile_time_arg_val(3);  // column tiles of qkvzab
    constexpr auto x_a = TensorAccessorArgs<4>();
    constexpr auto c_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t u1 = get_arg_val<uint32_t>(4);
    const uint32_t NR = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    constexpr uint32_t tb = 2048, SEG = W * 64;
    const auto x = TensorAccessor(x_a, x_addr, tb);
    const auto cr = TensorAccessor(c_a, c_addr, RB);
    const auto wt = TensorAccessor(w_a, w_addr, tb);
    const auto st = TensorAccessor(s_a, s_addr, RB);
    CircularBuffer cw(cb_w), cp(cb_prev), ct(cb_tin), cu(cb_u);
    cp.reserve_back(1);
    const uint32_t prev = cp.get_write_ptr();  // 3 rows x SEG
    uint32_t cur = 0xFFFFFFFF;
    auto read_block = [&](uint32_t c, uint32_t r) {
        ct.reserve_back(W);
        const uint32_t p = ct.get_write_ptr();
        for (uint32_t j = 0; j < W; j++) {
            noc_async_read(x.get_noc_addr(r * NQ + c * W + j, 0), p + j * tb, tb);
        }
        noc_async_read_barrier();
        ct.push_back(W);
    };
    for (uint32_t u = u0; u < u1; u++) {
        const uint32_t c = u / NR, r = u % NR;
        const bool extra = (r > 0) && (u == u0);
        if (extra) {
            read_block(c, r - 1);
        }
        read_block(c, r);
        if (c != cur) {
            cw.reserve_back(4 * W);
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
        if (extra) {
            cu.wait_front(W);
            noc_async_read(get_noc_addr(cu.get_read_ptr() + 29 * SEG), prev, 3 * SEG);
            noc_async_read_barrier();
            cu.pop_front(W);
        } else if (r == 0) {
            for (uint32_t i = 0; i < 3; i++) {
                noc_async_read(cr.get_noc_addr(i, off), prev + i * SEG, SEG);
            }
            noc_async_read_barrier();
        }
        cu.wait_front(W);
        const uint32_t b = cu.get_read_ptr();
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer cb(cb_rm + k);
            cb.reserve_back(W);
            const uint32_t d = cb.get_write_ptr();
            if (k < 3) {
                noc_async_read(get_noc_addr(prev + k * SEG), d, (3 - k) * SEG);
            }
            noc_async_read(get_noc_addr(b), d + (3 - k) * SEG, (29 + k) * SEG);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 4; k++) {
            CircularBuffer cb(cb_rm + k);
            cb.push_back(W);
        }
        noc_async_read(get_noc_addr(b + 29 * SEG), prev, 3 * SEG);  // this block's last 3 rows -> the next unit
        noc_async_read_barrier();
        if (r == NR - 1) {
            for (uint32_t i = 0; i < 3; i++) {
                noc_async_write(prev + i * SEG, st.get_noc_addr(i, off), SEG);
            }
            noc_async_write_barrier();
        }
        cu.pop_front(W);
    }
}
