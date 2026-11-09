// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3b (Exit the Moat, Episode 2): the DFlash2 verify window of one GDN layer as ONE data-movement program
// (DF_XWIN=1), instead of slice + 3 untilize + concat + tilize + copy: per channel tile c, X tile c =
// [hist rows 0..2; rows 0..T-1 of the in-projection qkvzab tile c; zeros], written to the L1 window Xl (read by the
// chained GDN program) and to the persistent DRAM window Xp (read by the commit). Whole tiles are read (hist from
// DRAM, qkvzab from L1), the rows moved inside L1. Pure copies: rows 0 .. 3 + T - 1 bit-identical to the concat.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#ifndef GDNW_XROW0
#define GDNW_XROW0 0
#endif

void kernel_main() {
    constexpr uint32_t T = GDNW_T;
    constexpr auto hista = TensorAccessorArgs<0>();
    constexpr auto qa = TensorAccessorArgs<hista.next_compile_time_args_offset()>();
    constexpr auto xla = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto xpa = TensorAccessorArgs<xla.next_compile_time_args_offset()>();
#ifdef GDNW_AIDX
    constexpr auto kva = TensorAccessorArgs<xpa.next_compile_time_args_offset()>();
#endif
    uint32_t i = 0;
    const uint32_t hist_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xl_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t hn = get_arg_val<uint32_t>(i++);

    constexpr uint32_t tb = 2048;
    const auto hist = TensorAccessor(hista, hist_addr, tb);
    const auto qz = TensorAccessor(qa, q_addr, tb);
    const auto xl = TensorAccessor(xla, xl_addr, tb);
    const auto xp = TensorAccessor(xpa, xp_addr, tb);
    CircularBuffer cb(0);
    cb.reserve_back(3);
    const uint32_t ph = cb.get_write_ptr(), pq = ph + tb, px = pq + tb;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t a = 255;  // >= T: the conv history from hist (after a commit / a new sequence)
#ifdef GDNW_AIDX
    {
        const auto kvt = TensorAccessor(kva, get_arg_val<uint32_t>(i++), (T * T + 16) * 4);
        noc_async_read(kvt.get_noc_addr(0, T * T * 4), px, 64);
        noc_async_read_barrier();
        a = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(px);
    }
#endif
    // ep2 t3b DF_NOCOMMIT: a < T: the history = rows a + 1 .. a + 3 of the previous window (this tile of Xp, read
    // before it is overwritten below)
    const uint32_t hrow = a < T ? a + 1 : 0;

    for (uint32_t c = h0; c < h0 + hn; c++) {
        if (a < T) {
            noc_async_read(xp.get_noc_addr(c), ph, tb);
        } else {
            noc_async_read(hist.get_noc_addr(c), ph, tb);
        }
        noc_async_read(qz.get_noc_addr(c), pq, tb);
        for (uint32_t q4 = 0; q4 < 4; q4++) {
            noc_async_read(zeros, px + q4 * 512, 512);
        }
        noc_async_read_barrier();
        for (uint32_t f = 0; f < 2; f++) {
            noc_async_read(get_noc_addr(ph + f * 512 + hrow * 32), px + f * 512, 3 * 32);
            // ms1o GDNW_XROW0: the stream's rows GDNW_XROW0 .. + T - 1 of the shared in-projection tile
            noc_async_read(get_noc_addr(pq + ((GDNW_XROW0) / 16) * 1024 + f * 512 + ((GDNW_XROW0) % 16) * 32),
                           px + f * 512 + 3 * 32, T * 32);
        }
        noc_async_read_barrier();
        noc_async_write(px, xl.get_noc_addr(c), tb);
        noc_async_write(px, xp.get_noc_addr(c), tb);
        noc_async_write_barrier();
    }
}
