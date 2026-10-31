// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: one-token residual add + RMSNorm, writer: core k unpacks its n output tiles c0 .. c0 + n of the
// normed row y (and, with WRITE_H, of the residual sum h) (row 0 of tile c = packed row c % 32 of packed tile c / 32),
// each built in L1 (MEM_ZEROS fill + two 32-byte chunks) and written whole. port5's rownorm writer, two outputs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_build = 4, cb_h = 6;

inline void zero_tiles(uint32_t b0, uint32_t n, uint32_t tb) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t i = 0; i < n; i++) {  // everything but row 0 of faces 0 / 1
        const uint32_t p = b0 + i * tb;
        noc_async_read(zeros, p + 32, 480);
        noc_async_read(zeros, p + 512 + 32, 480);
        noc_async_read(zeros, p + 1024, 512);
        noc_async_read(zeros, p + 1536, 512);
    }
}

inline void unpack_tiles(uint32_t src, uint32_t b0, uint32_t c0, uint32_t n, uint32_t tb) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = c0 + i, r = c % 32;
        const uint32_t s = src + (c / 32) * tb + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        noc_async_read(get_noc_addr(s), b0 + i * tb, 32);
        noc_async_read(get_noc_addr(s + 512), b0 + i * tb + 512, 32);
    }
}

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);  // output tiles of this core
    constexpr uint32_t WRITE_H = get_compile_time_arg_val(2);
    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto h_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    const auto h = TensorAccessor(h_a, h_addr, tb);
    CircularBuffer co(cb_out), cb(cb_build), ch(cb_h);
    cb.reserve_back((WRITE_H ? 2 : 1) * N);
    const uint32_t b0 = cb.get_write_ptr(), b1 = b0 + N * tb;
    zero_tiles(b0, (WRITE_H ? 2 : 1) * N, tb);
    if constexpr (WRITE_H) {
        ch.wait_front(NP);  // the residual sum is ready before the norm: write it first
        unpack_tiles(ch.get_read_ptr(), b1, c0, N, tb);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < N; i++) {
            noc_async_write(b1 + i * tb, h.get_noc_addr(c0 + i, 0), tb);
        }
    }
    co.wait_front(NP);
    unpack_tiles(co.get_read_ptr(), b0, c0, N, tb);
    noc_async_read_barrier();
    co.pop_front(NP);
    for (uint32_t i = 0; i < N; i++) {
        noc_async_write(b0 + i * tb, o.get_noc_addr(c0 + i, 0), tb);
    }
    noc_async_write_barrier();
}
