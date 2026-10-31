// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token RMSNorm, writer: core k unpacks its n output tiles c0 .. c0 + n of the [1, 1, 2048] row
// (row 0 of tile c = packed row c % 32 of packed tile c / 32), each built in L1 (MEM_ZEROS fill + two 32-byte chunks)
// and written whole.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_build = 4;

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);  // output tiles of this core
    constexpr auto o_a = TensorAccessorArgs<2>();
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out), cb(cb_build);
    cb.reserve_back(N);
    const uint32_t b0 = cb.get_write_ptr();
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t i = 0; i < N; i++) {  // everything but row 0 of faces 0 / 1
        const uint32_t p = b0 + i * tb;
        noc_async_read(zeros, p + 32, 480);
        noc_async_read(zeros, p + 512 + 32, 480);
        noc_async_read(zeros, p + 1024, 512);
        noc_async_read(zeros, p + 1536, 512);
    }
    co.wait_front(NP);
    const uint32_t src = co.get_read_ptr();
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t c = c0 + i, r = c % 32;
        const uint32_t s = src + (c / 32) * tb + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        noc_async_read(get_noc_addr(s), b0 + i * tb, 32);
        noc_async_read(get_noc_addr(s + 512), b0 + i * tb + 512, 32);
    }
    noc_async_read_barrier();
    co.pop_front(NP);
    for (uint32_t i = 0; i < N; i++) {
        noc_async_write(b0 + i * tb, o.get_noc_addr(c0 + i, 0), tb);
    }
    noc_async_write_barrier();
}
