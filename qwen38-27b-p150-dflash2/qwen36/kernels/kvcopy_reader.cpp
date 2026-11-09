// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 dfpc (Exit the Moat, Episode 2; DF_PREFIX_CACHE=1): copy a contiguous run of KV pages (tiles) inside each of the
// paged KV cache tensors: dst page d0 + t <- src page s0 + t, t = 0 .. n-1, for every buffer. Used to copy the KV
// blocks [0, b) of a cached prefix from the stream slot that holds them into another slot (a block of 64 tokens is a
// contiguous run of nkv * 2 * 4 tiles in a [blocks, nkv, 64, 128] TILE tensor). Byte copies, no math.
// The run (s0, d0, n) is read from a small int32 DRAM tensor (P), so ONE program serves every request (compiled at
// warm-up; nothing compiles or allocates at request time). All KV tensors share one accessor config (same shape,
// dtype, DRAM interleaved); their addresses and the page size are runtime args. Each core copies its share of t.
// Runtime args: core, ncores, page_bytes, p_addr, nbuf, addr[0 .. nbuf-1]. P words: [s0, d0, n, ...].

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

#ifndef KVC_BATCH
#define KVC_BATCH 8
#endif

void kernel_main() {
    constexpr auto kva = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t core = get_arg_val<uint32_t>(i++);
    const uint32_t ncores = get_arg_val<uint32_t>(i++);
    const uint32_t pb = get_arg_val<uint32_t>(i++);
    const uint32_t p_addr = get_arg_val<uint32_t>(i++);
    const uint32_t nbuf = get_arg_val<uint32_t>(i++);

    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t l1 = cb.get_write_ptr();  // KVC_BATCH pages + 64 bytes for P
    const uint32_t lp = l1 + KVC_BATCH * pb;
    {
        const auto pt = TensorAccessor(pa, p_addr, 64);
        noc_async_read(pt.get_noc_addr(0), lp, 64);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* P = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lp);
    const uint32_t s0 = P[0], d0 = P[1], n = P[2];
    const uint32_t per = (n + ncores - 1) / ncores;
    const uint32_t lo = core * per < n ? core * per : n;
    const uint32_t hi = lo + per < n ? lo + per : n;
    for (uint32_t b = 0; b < nbuf; b++) {
        const auto t = TensorAccessor(kva, get_arg_val<uint32_t>(i + b), pb);
        for (uint32_t t0 = lo; t0 < hi; t0 += KVC_BATCH) {
            const uint32_t m = hi - t0 < KVC_BATCH ? hi - t0 : KVC_BATCH;
            for (uint32_t j = 0; j < m; j++) {
                noc_async_read(t.get_noc_addr(s0 + t0 + j), l1 + j * pb, pb);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < m; j++) {
                noc_async_write(l1 + j * pb, t.get_noc_addr(d0 + t0 + j), pb);
            }
            noc_async_write_barrier();
        }
    }
}
