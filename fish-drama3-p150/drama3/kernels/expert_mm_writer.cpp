// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token indexed expert matmul, writer: streams the odd K tiles of each job's weights (cb 4; the
// reader takes the even ones), then writes the job's output tile (job j * NT + n -> page j * NT + n of the compact
// [1, k, 1, N] result).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_w2 = 4, cb_ids2 = 5, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t KB = get_compile_time_arg_val(2);
    constexpr uint32_t NSLOT = get_compile_time_arg_val(3);
    constexpr auto w_a = TensorAccessorArgs<4>();
    constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto o_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t nj = get_arg_val<uint32_t>(4);
    const uint32_t tw = get_tile_size(cb_w2), tb = get_tile_size(cb_out);
    const auto w = TensorAccessor(w_a, w_addr, tw);
    const auto ids = TensorAccessor(i_a, i_addr, 2 * NSLOT);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    Noc noc;
    CircularBuffer cw(cb_w2), ci(cb_ids2), co(cb_out);
    ci.reserve_back(1);
    noc_async_read(ids.get_noc_addr(0, 0), ci.get_write_ptr(), 2 * NSLOT);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ci.get_write_ptr());
    for (uint32_t q = 0; q < nj; q++) {
        const uint32_t job = j0 + q, j = job / NT, n = job % NT;
        const uint32_t e = id[j];
        for (uint32_t k0 = 0; k0 < KT; k0 += KB) {
            cw.reserve_back(KB / 2);
            for (uint32_t k = 0; k < KB / 2; k++) {
                noc.async_read(w, cw, tw, {.page_id = (e * KT + k0 + 2 * k + 1) * NT + n}, {.offset_bytes = k * tw});
            }
            noc.async_read_barrier();
            cw.push_back(KB / 2);
        }
        co.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(co), o, tb, {.offset_bytes = 0}, {.page_id = job});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}
