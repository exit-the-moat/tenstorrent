// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 dfpc (Exit the Moat, Episode 2; DF_PREFIX_CACHE=1, check only): byte compare of two runs of KV pages in each
// paged KV tensor: page s0 + t against page d0 + t, t = 0 .. n-1 (the run the copy program just wrote). Counts the
// 32-bit words that differ; each core writes [mismatching words, pages compared] to its own 64-byte row of the output
// int32 tensor O. Same parameter tensor P as kvcopy_reader.cpp ([s0, d0, n]), so one program, compiled at warm-up.
// Runtime args: core, ncores, page_bytes, p_addr, o_addr, nbuf, addr[0 .. nbuf-1].

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto kva = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t core = get_arg_val<uint32_t>(i++);
    const uint32_t ncores = get_arg_val<uint32_t>(i++);
    const uint32_t pb = get_arg_val<uint32_t>(i++);
    const uint32_t p_addr = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t nbuf = get_arg_val<uint32_t>(i++);

    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t la = cb.get_write_ptr();  // page a, page b, 64 bytes P, 64 bytes out
    const uint32_t lb = la + pb;
    const uint32_t lp = lb + pb;
    const uint32_t lo = lp + 64;
    {
        const auto pt = TensorAccessor(pa, p_addr, 64);
        noc_async_read(pt.get_noc_addr(0), lp, 64);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* P = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lp);
    const uint32_t s0 = P[0], d0 = P[1], n = P[2];
    const uint32_t per = (n + ncores - 1) / ncores;
    const uint32_t t_lo = core * per < n ? core * per : n;
    const uint32_t t_hi = t_lo + per < n ? t_lo + per : n;
    uint32_t bad = 0, pages = 0;
    for (uint32_t b = 0; b < nbuf; b++) {
        const auto t = TensorAccessor(kva, get_arg_val<uint32_t>(i + b), pb);
        for (uint32_t x = t_lo; x < t_hi; x++) {
            noc_async_read(t.get_noc_addr(s0 + x), la, pb);
            noc_async_read(t.get_noc_addr(d0 + x), lb, pb);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(la);
            volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lb);
            for (uint32_t w = 0; w < pb / 4; w++) {
                bad += A[w] != B[w];
            }
            pages++;
        }
    }
    volatile tt_l1_ptr uint32_t* O = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lo);
    for (uint32_t w = 0; w < 16; w++) {
        O[w] = 0;
    }
    O[0] = bad;
    O[1] = pages;
    const auto ot = TensorAccessor(oa, o_addr, 64 * ncores);
    noc_async_write(lo, ot.get_noc_addr(0, core * 64), 64);
    noc_async_write_barrier();
}
