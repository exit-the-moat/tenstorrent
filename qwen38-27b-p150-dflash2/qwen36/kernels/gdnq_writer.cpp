// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused one-token GDN decode (TP qwen36), writer. Per sequence r of this core's row pair: the new fp32 state of
// (r, hv) in place, then the gated output row (row 0 of the 4 output tiles of head hv) into row r of the [1, B, NV*128]
// L1 output. Inactive row r0 + 1 and (last pair) rows r0 + 2 .. 31 of the head's output tiles are written as zeros.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_state = 17;

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr uint32_t STF32 = get_compile_time_arg_val(1);
    constexpr auto reca = TensorAccessorArgs<2>();
    constexpr auto outa = TensorAccessorArgs<reca.next_compile_time_args_offset()>();

    const uint32_t hv = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    const uint32_t B = get_arg_val<uint32_t>(2);
    const uint32_t last = get_arg_val<uint32_t>(3);
    const uint32_t rec_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048;
    const auto rec = TensorAccessor(reca, rec_addr, tf);
    const auto out = TensorAccessor(outa, out_addr, tb);
    CircularBuffer cst(cb_state), cout(cb_out);
#ifdef GDNQ_TS
    CircularBuffer cts(27);
    cts.wait_front(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_read_ptr());
#define WTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    WTS(48);
#else
#define WTS(k)
#endif

    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t r = r0 + s;
        cst.wait_front(16);
        const uint32_t ps = cst.get_read_ptr();
#ifndef GDNQ_NOSTATE
        for (uint32_t t = 0; t < 16; t++) {
            noc_async_write(ps + t * tf, rec.get_noc_addr((r * NV + hv) * 16 + t), tf);
        }
#endif
        noc_async_write_barrier();
        WTS(49 + 3 * s);
        cst.pop_front(16);
        cout.wait_front(4);
        const uint32_t po = cout.get_read_ptr();
        for (uint32_t k = 0; k < 4; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(po + k * tb + f * 512, out.get_noc_addr(4 * hv + k, f * 512 + r * 32), 32);
            }
        }
        WTS(50 + 3 * s);
        noc_async_write_barrier();
        WTS(51 + 3 * s);
        cout.pop_front(4);
    }
    for (uint32_t k = 0; k < 4; k++) {
        const uint32_t pg = 4 * hv + k;
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
        const auto tsd = TensorAccessor(tsa, get_arg_val<uint32_t>(6), 2048);
        noc_async_write(cts.get_read_ptr(), tsd.get_noc_addr(get_arg_val<uint32_t>(7)), 2048);
        noc_async_write_barrier();
    }
#endif
}
