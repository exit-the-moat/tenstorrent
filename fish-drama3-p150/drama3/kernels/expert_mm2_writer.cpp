// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: ExpertMM v2 writer (deep weight stream as expert_mm2_reader.cpp; the outputs are written at
// the end, so the stream never waits on the compute). port5: streams the odd K tiles of each job's weights (cb 4; the
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
    constexpr uint32_t D = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr auto w_a = TensorAccessorArgs<6>();
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
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nb = nj * NKB;
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, job = j0 + q, j = job / NT, n = job % NT;
            const uint32_t e = id[j];
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k + 1) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
    co.wait_front(nj);  // the output CB holds every job of this core
    for (uint32_t q = 0; q < nj; q++) {
        noc_async_write(co.get_read_ptr() + q * tb, o.get_noc_addr(j0 + q), tb);
    }
    noc_async_write_barrier();
    co.pop_front(nj);
}
