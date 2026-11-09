// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn: fused one-token GDN decode v2 (QWEN36_GDN_FAST=1), writer. Per sequence r of this core's row pair:
//   CS > 1: the own o tiles (cb_o tiles c0 .. c0 + NC, the same L1 address on every core) to the other CS - 1 cores
//           of the head, then semaphore 1 there; wait until all of theirs arrived here, release the compute (cb_orecv)
//   the new state tiles (own columns) in place, the gated output rows of the own column tiles into row r of the
//   [1, B, NV*128] L1 output (inactive rows as zeros, as gdnq_writer.cpp).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_state = 17, cb_o = 28, cb_osig = 32, cb_orecv = 33;

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr uint32_t STF32 = get_compile_time_arg_val(1);
    constexpr uint32_t CS = get_compile_time_arg_val(2);
    constexpr uint32_t SEMO = get_compile_time_arg_val(3);
    constexpr uint32_t NC = 4 / CS;
    constexpr auto reca = TensorAccessorArgs<4>();
    constexpr auto outa = TensorAccessorArgs<reca.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t p = get_arg_val<uint32_t>(i++);
    const uint32_t B = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t last = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    uint32_t peer[2 * (CS > 1 ? CS - 1 : 1)];
    for (uint32_t k = 0; k < 2 * (CS - 1); k++) {
        peer[k] = get_arg_val<uint32_t>(i++);
    }
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;
    const uint32_t c0 = ci * NC;

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048, to = 4096;
    const auto rec = TensorAccessor(reca, rec_addr, tf);
    const auto out = TensorAccessor(outa, out_addr, tb);
    CircularBuffer cst(cb_state), cout(cb_out), co(cb_o), cos(cb_osig), cor(cb_orecv);
#ifdef GDNQ_TS
    CircularBuffer cts(27);
    cts.wait_front(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_read_ptr());
#define WTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    WTS(48);
#else
#define WTS(k)
#endif
    volatile tt_l1_ptr uint32_t* semo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEMO));
    const uint32_t obase = co.get_write_ptr();  // never moves on this RISC: cb_o's base, slot s at + s * 4 tiles

    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t r = r0 + s;
        if constexpr (CS > 1) {
            cos.wait_front(1);
            const uint32_t src = obase + (s * 4 + c0) * to;
            for (uint32_t k = 0; k < CS - 1; k++) {
                noc_async_write(src, get_noc_addr(peer[2 * k], peer[2 * k + 1], src), NC * to);
            }
            noc_async_write_barrier();
            for (uint32_t k = 0; k < CS - 1; k++) {
                noc_semaphore_inc(get_noc_addr(peer[2 * k], peer[2 * k + 1], get_semaphore(SEMO)), 1);
            }
            cos.pop_front(1);
            WTS(49 + 4 * s);
        }
        cst.wait_front(4 * NC);
        const uint32_t ps = cst.get_read_ptr();
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_write(ps + (ki * NC + j) * tf, rec.get_noc_addr((r * NV + hv) * 16 + ki * 4 + c0 + j), tf);
            }
        }
        if constexpr (CS > 1) {
            noc_semaphore_wait_min(semo, (CS - 1) * (s + 1));
            cor.reserve_back(1);
            cor.push_back(1);
            WTS(50 + 4 * s);
        }
        noc_async_write_barrier();
        cst.pop_front(4 * NC);
        cout.wait_front(NC);
        WTS(51 + 4 * s);
        const uint32_t po = cout.get_read_ptr();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(po + k * tb + f * 512, out.get_noc_addr(4 * hv + c0 + k, f * 512 + r * 32), 32);
            }
        }
        noc_async_write_barrier();
        cout.pop_front(NC);
        WTS(52 + 4 * s);
    }
    for (uint32_t k = 0; k < NC; k++) {
        const uint32_t pg = 4 * hv + c0 + k;
        if (ns == 1) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, f * 512 + (r0 + 1) * 32), 32);
            }
        }
        if (last) {
            const uint32_t rz = r0 + 2;
            if (rz < 16) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, f * 512 + rz * 32), (16 - rz) * 32);
                }
            }
            noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1024), 512);
            noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1536), 512);
        }
    }
    noc_async_write_barrier();
#ifdef GDNQ_TS
    WTS(63);
    {
        constexpr auto tsa = TensorAccessorArgs<outa.next_compile_time_args_offset()>();
        const auto tsd = TensorAccessor(tsa, get_arg_val<uint32_t>(i), 2048);
        noc_async_write(cts.get_read_ptr(), tsd.get_noc_addr(get_arg_val<uint32_t>(i + 1)), 2048);
        noc_async_write_barrier();
    }
#endif
}
