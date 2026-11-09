// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn: fused one-token GDN decode v4 (QWEN36_GDN_FAST=4), writer. First (this RISC is idle until the o tiles):
// the gate inputs a / b (as full tiles, every element the scalar: the value in 64 bytes, then doubling local copies)
// and the dt_bias / A_neg tiles of the head, so the compute does the gates while the reader brings the conv rows.
// Then, as gdnq2_writer.cpp, per sequence r of this core's row pair:
//   CS > 1: the own o tiles (cb_o tiles c0 .. c0 + NC, the same L1 address on every core) to the other CS - 1 cores
//           of the head, then semaphore 1 there; wait until all of theirs arrived here, release the compute (cb_orecv)
//   the new state tiles (own columns) in place, the gated output rows of the own column tiles into row r of the
//   [1, B, NV*128] L1 output (inactive rows as zeros, as gdnq_writer.cpp).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_scr = 10, cb_out = 16, cb_state = 17, cb_o = 28,
                   cb_osig = 32, cb_orecv = 33;

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr uint32_t STF32 = get_compile_time_arg_val(1);
    constexpr uint32_t CS = get_compile_time_arg_val(2);
    constexpr uint32_t SEMO = get_compile_time_arg_val(3);
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);
    constexpr uint32_t NC = 4 / CS;
#ifdef GDNQ_XT
    constexpr uint32_t XT = GDNQ_XT;  // r64e: qkvzab width in tiles (second tile row of a B > 32 step)
#else
    constexpr uint32_t XT = 1;
#endif
    constexpr uint32_t OT = 4 * NV;   // r64e: gated-output [1,B,Nv*128] tiles per row tile
    constexpr auto reca = TensorAccessorArgs<6>();
    constexpr auto outa = TensorAccessorArgs<reca.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<outa.next_compile_time_args_offset()>();
    constexpr auto dtba = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto anega = TensorAccessorArgs<dtba.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t p = get_arg_val<uint32_t>(i++);
    const uint32_t B = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t last = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    uint32_t peer[2 * (CS > 1 ? CS - 1 : 1)];
    for (uint32_t k = 0; k < 2 * (CS - 1); k++) {
        peer[k] = get_arg_val<uint32_t>(i++);
    }
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;
    const uint32_t c0 = ci * NC;
    // t3c: row r of a tile at RO(r): faces 0 / 1 (r < 16) or 2 / 3 (r >= 16; batches above 16)
    // r64e: r is taken inside its tile row (r % 32); rows >= 32 are one tile row further, i.e. their page index
    // gains (r / 32) * <tiles per row tile> (OT for the gated output, XT for qkvzab).
    auto RO = [](uint32_t r) -> uint32_t { return ((r % 32) / 16) * 1024 + ((r % 32) % 16) * 32; };
    auto OP = [](uint32_t r) -> uint32_t { return (r / 32) * OT; };   // gated-output page offset
    auto XP = [](uint32_t r) -> uint32_t { return (r / 32) * XT; };   // qkvzab page offset
#ifdef GDNQ_XROW
    const uint32_t xr0 = GDNQ_XROW;  // ep2 t3 (DFlash2 chained verify): gate inputs from / output into row GDNQ_XROW
#else
    const uint32_t xr0 = r0;
#endif

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
    {
        // ---- the gate inputs
        Noc noc;
        const auto xq = TensorAccessor(xa, x_addr, tb);
        const auto dtb = TensorAccessor(dtba, dtb_addr, tb);
        const auto aneg = TensorAccessor(anega, aneg_addr, tb);
        CircularBuffer cdt(cb_dtb), can(cb_aneg), scr(cb_scr), cav(cb_a), cbv(cb_b);
        cdt.reserve_back(1);
        can.reserve_back(1);
        scr.reserve_back(1);
        cav.reserve_back(ns);
        cbv.reserve_back(ns);
        noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
        noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
        const uint32_t ps = scr.get_write_ptr();
        const uint32_t ca = ACOL + hv, cbb = BCOL + hv;
        for (uint32_t s = 0; s < ns; s++) {
            const uint32_t r = xr0 + s;
            noc_async_read(xq.get_noc_addr(ca / 32 + XP(r), ((ca % 32) / 16) * 512 + RO(r)), ps + s * 128, 32);
            noc_async_read(xq.get_noc_addr(cbb / 32 + XP(r), ((cbb % 32) / 16) * 512 + RO(r)), ps + s * 128 + 64, 32);
        }
        noc.async_read_barrier();
        for (uint32_t s = 0; s < ns; s++) {
            volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps + s * 128);
            const uint32_t va = row[ca % 16], vb = row[32 + cbb % 16];
            volatile tt_l1_ptr uint32_t* ta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cav.get_write_ptr() + s * tb);
            volatile tt_l1_ptr uint32_t* tbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbv.get_write_ptr() + s * tb);
            const uint32_t wa = va | (va << 16), wb = vb | (vb << 16);
            for (uint32_t w = 0; w < 16; w++) {
                ta[w] = wa;
                tbp[w] = wb;
            }
        }
        for (uint32_t sz = 64; sz < tb; sz *= 2) {
            for (uint32_t s = 0; s < ns; s++) {
                const uint32_t ta = cav.get_write_ptr() + s * tb, tbb = cbv.get_write_ptr() + s * tb;
                noc_async_read(get_noc_addr(ta), ta + sz, sz);
                noc_async_read(get_noc_addr(tbb), tbb + sz, sz);
            }
            noc.async_read_barrier();
        }
        cdt.push_back(1);
        can.push_back(1);
        cav.push_back(ns);
        cbv.push_back(ns);
    }
    // the rows of the output tiles that are not an active sequence: zeros, first (nothing depends on them)
    for (uint32_t k = 0; k < NC; k++) {
        const uint32_t pg = 4 * hv + c0 + k;
        if (ns == 1) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg + OP(xr0 + 1), f * 512 + RO(xr0 + 1)), 32);
            }
        }
        if (last) {
            const uint32_t rz = xr0 + 2;
            // r64e: rows below the last active pair only; rz == B (the last pair of a full batch) writes nothing,
            // exactly as before this patch (a page index past the tensor would land in unrelated memory)
            if (rz < 16) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, f * 512 + rz * 32), (16 - rz) * 32);
                }
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1024), 512);
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1536), 512);
            } else if (rz < 32) {  // t3c: rows rz .. 31 of faces 2 / 3
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1024 + f * 512 + (rz - 16) * 32), (32 - rz) * 32);
                }
            }
        }
    }
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
        if constexpr (CS > 1) {  // release the rms norm as soon as the other columns are here (before the state)
            noc_semaphore_wait_min(semo, (CS - 1) * (s + 1));
            cor.reserve_back(1);
            cor.push_back(1);
            WTS(50 + 4 * s);
        }
        cst.wait_front(4 * NC);
        const uint32_t ps = cst.get_read_ptr();
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_write(ps + (ki * NC + j) * tf, rec.get_noc_addr((r * NV + hv) * 16 + ki * 4 + c0 + j), tf);
            }
        }
        noc_async_write_barrier();
        cst.pop_front(4 * NC);
        cout.wait_front(NC);
        WTS(51 + 4 * s);
        const uint32_t po = cout.get_read_ptr();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(po + k * tb + f * 512,
                                out.get_noc_addr(4 * hv + c0 + k + OP(xr0 + s), f * 512 + RO(xr0 + s)), 32);
            }
        }
        noc_async_write_barrier();
        cout.pop_front(NC);
        WTS(52 + 4 * s);
    }
    noc_async_write_barrier();
#ifdef GDNQ_TS
    WTS(63);
    {
        constexpr auto tsa = TensorAccessorArgs<anega.next_compile_time_args_offset()>();
        const auto tsd = TensorAccessor(tsa, get_arg_val<uint32_t>(i), 2048);
        noc_async_write(cts.get_read_ptr(), tsd.get_noc_addr(get_arg_val<uint32_t>(i + 1)), 2048);
        noc_async_write_barrier();
    }
#endif
}
