// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: TP decode residual add + RMSNorm in one program (kernels/addnt_reader.cpp), writer of core k: sends row 0 of
// this core's partial tile P (fp32: two 64-byte face rows) to row k of G on every core (unicast, + a semaphore
// increment there; rows NC.. of G are zeroed locally), writes the new residual H into its own shard, then waits for the
// NC partials, hands G to compute and writes its PER output tiles of Y to the interleaved output (L1 or DRAM).
// (A gather-to-core-0 + multicast variant hung the device once, 2026-11-03 00:13: unicast only.)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_h = 4, cb_p = 6, cb_g = 7, cb_out = 16;
constexpr uint32_t SEM = 0;

// row k of G: tile k / 32, row k % 32 (fp32 tile: faces of 1024 bytes, face rows of 64 bytes)
inline uint32_t g_row(uint32_t k) {
    const uint32_t r = k % 32;
    return (k / 32) * 4096 + (r < 16 ? r * 64 : 2048 + (r - 16) * 64);
}

void kernel_main() {
    constexpr uint32_t PER = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);  // cores
    constexpr uint32_t NG = (NC + 31) / 32;               // G tiles
    constexpr uint32_t WRITE_H = get_compile_time_arg_val(2);
    constexpr uint32_t DBG = get_compile_time_arg_val(3);  // bit 0: no exchange (timing only, wrong values)
    constexpr auto o_a = TensorAccessorArgs<4>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = k * PER;
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out), ch(cb_h), cg(cb_g), cp(cb_p);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    cg.reserve_back(NG);
    const uint32_t g0 = cg.get_write_ptr();
    if constexpr (NC < NG * 32) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t j = NC; j < NG * 32; j++) {
            noc_async_read(zeros, g0 + g_row(j), 64);
            noc_async_read(zeros, g0 + g_row(j) + 1024, 64);
        }
    }
    cp.wait_front(1);
    const uint32_t t0 = cp.get_read_ptr();
    const uint32_t gk = g0 + g_row(k);
    constexpr uint32_t NX = (DBG & 1) ? 1 : NC;
    for (uint32_t j = 0; j < NX; j++) {
        const uint64_t a = get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j), gk);
        noc_async_write(t0, a, 64);
        noc_async_write(t0 + 1024, a + 1024, 64);
    }
    noc_async_write_barrier();
    if constexpr (!(DBG & 1)) {
        for (uint32_t j = 0; j < NC; j++) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j),
                                           get_semaphore(SEM)), 1);
        }
    }
    cp.pop_front(1);
    if constexpr (WRITE_H) {  // the residual sum is ready before the norm: write it while the partials come in
        ch.wait_front(PER);
        noc_async_write(ch.get_read_ptr(), get_noc_addr(h_addr), PER * tb);
    }
    noc_async_read_barrier();  // G's zero fill
    if constexpr (!(DBG & 1)) {
        noc_semaphore_wait(sem, NC);
        noc_semaphore_set(sem, 0);
    }
    cg.push_back(NG);
    co.wait_front(PER);
    const uint32_t r0 = co.get_read_ptr();
    for (uint32_t i = 0; i < PER; i++) {
        noc_async_write(r0 + i * tb, o.get_noc_addr(c0 + i, 0), tb);
    }
    noc_async_write_barrier();
    co.pop_front(PER);
    noc_async_atomic_barrier();
}
