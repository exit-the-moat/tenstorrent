// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: one-token residual add + RMSNorm over NC cores (port of drama3/kernels/addnorm2_writer.cpp), writer of core k:
// gathers the fp32 partial sums of squares (row 0 of faces 0 / 1 of every core's T tile goes to row k of G on every
// core, then a semaphore increment there; G's other rows are zeroed locally, nobody else writes them), then unpacks its
// PER output tiles of y (and, with WRITE_H, of the residual sum h): row 0 of tile c0 + i = packed row i, each tile built
// in L1 (MEM_ZEROS fill + two 32-byte chunks) and written whole.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_build = 4, cb_h = 6, cb_g = 7, cb_out = 16, cb_t = 27;
constexpr uint32_t SEM = 0;

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

inline void unpack_tiles(uint32_t src, uint32_t b0, uint32_t n, uint32_t tb) {
    for (uint32_t i = 0; i < n; i++) {
        noc_async_read(get_noc_addr(src + i * 32), b0 + i * tb, 32);
        noc_async_read(get_noc_addr(src + 512 + i * 32), b0 + i * tb + 512, 32);
    }
}

void kernel_main() {
    constexpr uint32_t PER = get_compile_time_arg_val(0);  // output tiles of this core
    constexpr uint32_t NC = get_compile_time_arg_val(1);   // cores, <= 16
    constexpr uint32_t WRITE_H = get_compile_time_arg_val(2);
    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto h_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = k * PER;
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    const auto h = TensorAccessor(h_a, h_addr, tb);
    CircularBuffer co(cb_out), cb(cb_build), ch(cb_h), cg(cb_g), ct(cb_t);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    cg.reserve_back(1);
    const uint32_t g0 = cg.get_write_ptr();  // fp32 tile: face = 1024 bytes, face row = 64 bytes
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        if (NC < 16) {
            noc_async_read(zeros, g0 + NC * 64, (16 - NC) * 64 > 512 ? 512 : (16 - NC) * 64);
            noc_async_read(zeros, g0 + 1024 + NC * 64, (16 - NC) * 64 > 512 ? 512 : (16 - NC) * 64);
            if ((16 - NC) * 64 > 512) {
                noc_async_read(zeros, g0 + NC * 64 + 512, (16 - NC) * 64 - 512);
                noc_async_read(zeros, g0 + 1024 + NC * 64 + 512, (16 - NC) * 64 - 512);
            }
        }
        for (uint32_t q = 2048; q < 4096; q += 512) {
            noc_async_read(zeros, g0 + q, 512);
        }
    }
    cb.reserve_back((WRITE_H ? 2 : 1) * PER);
    const uint32_t b0 = cb.get_write_ptr(), b1 = b0 + PER * tb;
    zero_tiles(b0, (WRITE_H ? 2 : 1) * PER, tb);
    ct.wait_front(1);
    const uint32_t t0 = ct.get_read_ptr();
    for (uint32_t j = 0; j < NC; j++) {
        const uint32_t nx = get_arg_val<uint32_t>(3 + 2 * j), ny = get_arg_val<uint32_t>(4 + 2 * j);
        noc_async_write(t0, get_noc_addr(nx, ny, g0 + k * 64), 64);
        noc_async_write(t0 + 1024, get_noc_addr(nx, ny, g0 + 1024 + k * 64), 64);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < NC; j++) {
        const uint32_t nx = get_arg_val<uint32_t>(3 + 2 * j), ny = get_arg_val<uint32_t>(4 + 2 * j);
        noc_semaphore_inc(get_noc_addr(nx, ny, get_semaphore(SEM)), 1);
    }
    ct.pop_front(1);
    if constexpr (WRITE_H) {  // the residual sum is ready before the norm: write it while the partials come in
        ch.wait_front(1);
        unpack_tiles(ch.get_read_ptr(), b1, PER, tb);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < PER; i++) {
            noc_async_write(b1 + i * tb, h.get_noc_addr(c0 + i, 0), tb);
        }
    }
    noc_async_read_barrier();  // G's zero fill
    noc_semaphore_wait(sem, NC);
    noc_semaphore_set(sem, 0);
    cg.push_back(1);
    co.wait_front(1);
    unpack_tiles(co.get_read_ptr(), b0, PER, tb);
    noc_async_read_barrier();
    co.pop_front(1);
    for (uint32_t i = 0; i < PER; i++) {
        noc_async_write(b0 + i * tb, o.get_noc_addr(c0 + i, 0), tb);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
