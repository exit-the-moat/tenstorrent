// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn (Exit the Moat, Episode 2): fused one-token GDN decode v2 (QWEN36_GDN_FAST=1), reader. Same math and data
// as gdnq_reader.cpp, two changes:
//   - every read that does not depend on another core (conv rows, taps, selectors, norm weight, dt_bias / A_neg, z,
//     a / b, the state) is issued at once behind ONE barrier (v1: three serial batches, the constants only after the
//     conv-shift writes); the a / b full tiles are filled by doubling local copies instead of 1024 CPU stores;
//   - CS cores per (value head, row pair): core ci owns the NC = 4 / CS state column tiles [c0, c0 + NC) (c0 = ci NC)
//     of the [128 k, 128 v] state, the matching v conv channels and z tiles. q / k are computed by every core of the
//     head (bit-identical copies); the o tiles are exchanged by the writers for the rms norm (gdnq2_writer.cpp).
// Conv rows of the shared q / k channels: all RF * CS cores of key head hk read them, the owner (hv % RF == 0,
// ci == 0) rewrites them after the others signal (semaphore 0). v channel rows: one owner each, no semaphore.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_sdst = 8,
                   cb_onw = 9, cb_scr = 10, cb_sel = 11, cb_stage = 12, cb_smm = 13;

void kernel_main() {
    constexpr uint32_t SEM = get_compile_time_arg_val(0);
    constexpr uint32_t NK = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t ZT0 = get_compile_time_arg_val(3);
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);
    constexpr uint32_t CMODE = get_compile_time_arg_val(6);
    constexpr uint32_t STF32 = get_compile_time_arg_val(7);
    constexpr uint32_t CS = get_compile_time_arg_val(8);  // cores per (head, pair): 1, 2 or 4
    constexpr uint32_t NC = 4 / CS;                       // state column tiles per core
    constexpr uint32_t RF = NV / NK;
    constexpr auto xa = TensorAccessorArgs<9>();
    constexpr auto s0a = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto s1a = TensorAccessorArgs<s0a.next_compile_time_args_offset()>();
    constexpr auto s2a = TensorAccessorArgs<s1a.next_compile_time_args_offset()>();
    constexpr auto s3a = TensorAccessorArgs<s2a.next_compile_time_args_offset()>();
    constexpr auto wpa = TensorAccessorArgs<s3a.next_compile_time_args_offset()>();
    constexpr auto sela = TensorAccessorArgs<wpa.next_compile_time_args_offset()>();
    constexpr auto reca = TensorAccessorArgs<sela.next_compile_time_args_offset()>();
    constexpr auto dtba = TensorAccessorArgs<reca.next_compile_time_args_offset()>();
    constexpr auto anega = TensorAccessorArgs<dtba.next_compile_time_args_offset()>();
    constexpr auto onwa = TensorAccessorArgs<anega.next_compile_time_args_offset()>();
    constexpr uint32_t KT0 = 4 * NK, VT0 = 8 * NK;

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t p = get_arg_val<uint32_t>(i++);
    const uint32_t B = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t sel_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);
    const uint32_t owner_x = get_arg_val<uint32_t>(i++);
    const uint32_t owner_y = get_arg_val<uint32_t>(i++);

    const uint32_t hk = hv / RF;
    const uint32_t c0 = ci * NC;
    const bool owner = (hv % RF) == 0 && ci == 0;
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;
    const uint32_t roff = r0 * 32;

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048;
    const auto xq = TensorAccessor(xa, x_addr, tb);
    const auto st0 = TensorAccessor(s0a, s0_addr, tb);
    const auto st1 = TensorAccessor(s1a, s1_addr, tb);
    const auto st2 = TensorAccessor(s2a, s2_addr, tb);
    const auto st3 = TensorAccessor(s3a, s3_addr, tb);
    const auto wp = TensorAccessor(wpa, wp_addr, tb);
    const auto sel = TensorAccessor(sela, sel_addr, tb);
    const auto rec = TensorAccessor(reca, rec_addr, tf);
    const auto dtb = TensorAccessor(dtba, dtb_addr, tb);
    const auto aneg = TensorAccessor(anega, aneg_addr, tb);
    const auto onw = TensorAccessor(onwa, onw_addr, tb);

    auto chan = [&](uint32_t t) -> uint32_t {
        return t < 4 ? 4 * hk + t : (t < 8 ? KT0 + 4 * hk + (t - 4) : VT0 + 4 * hv + (t - 8));
    };
    // packed rows this core reads (and, for v, rewrites): q 0..3, k 4..7, v 8 + [c0, c0 + NC)
    auto mine = [&](uint32_t t) -> bool { return t < 8 || (t - 8 >= c0 && t - 8 < c0 + NC); };

    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
#ifdef GDNQ_TS
    // ablation: per-core stage timestamps in one shared tile of CB 27 (reader 0..15, compute 16..47, writer 48..63)
    CircularBuffer cts(27);
    cts.reserve_back(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_write_ptr());
#define RTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    RTS(0);
    cts.push_back(1);
#else
#define RTS(k)
#endif

    // ---- every independent read at once
    CircularBuffer stg(cb_stage), cx(cb_x), cq(cb_q);
    stg.reserve_back(3);
    cx.reserve_back(ns);
    cq.reserve_back(3 * ns);
    const uint32_t sb = stg.get_write_ptr(), px = cx.get_write_ptr(), pq = cq.get_write_ptr();
    // the state of sequence r0 (the largest transfer) first, two copies (matmul operand, SFPU copy): tile ki * NC + j
    CircularBuffer csd(cb_sdst), csm(cb_smm);
    csd.reserve_back(4 * NC);
    csm.reserve_back(4 * NC);
    auto read_state = [&](uint32_t r) {
        const uint32_t pd = csd.get_write_ptr(), pm = csm.get_write_ptr();
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                const uint64_t a = rec.get_noc_addr((r * NV + hv) * 16 + ki * 4 + c0 + j);
                noc_async_read(a, pm + (ki * NC + j) * tf, tf);
                noc_async_read(a, pd + (ki * NC + j) * tf, tf);
            }
        }
    };
    read_state(r0);
    // conv inputs: chunk (m, t, f) -> staging slot (m * 24 + 2t + f) * 64
    for (uint32_t t = 0; t < 12; t++) {
        if (!mine(t)) {
            continue;
        }
        const uint32_t c = chan(t);
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512 + roff, sl = (2 * t + f) * 64;
            noc_async_read(xq.get_noc_addr(c, off), sb + sl, 64);
            if constexpr (CMODE == 2) {
                noc_async_read(st1.get_noc_addr(c, f * 512), sb + 24 * 64 + sl, 64);
                noc_async_read(st1.get_noc_addr(c, f * 512 + 64), sb + 48 * 64 + sl, 64);
            } else {
                noc_async_read(st1.get_noc_addr(c, off), sb + 24 * 64 + sl, 64);
                noc_async_read(st2.get_noc_addr(c, off), sb + 48 * 64 + sl, 64);
                noc_async_read(st3.get_noc_addr(c, off), sb + 72 * 64 + sl, 64);
            }
        }
    }
    // packed tiles: rows 12..15 (with NC < 4 also the v rows 8..11, the own ones are filled below) of faces 0 / 1
    // and faces 2 / 3 are zero
    constexpr uint32_t zrow = NC == 4 ? 12 : 8;
    for (uint32_t k = 0; k < 4 * ns; k++) {
        const uint32_t pt = k < ns ? px + k * tb : pq + (k - ns) * tb;
        noc_async_read(zeros, pt + zrow * 32, (16 - zrow) * 32);
        noc_async_read(zeros, pt + 512 + zrow * 32, (16 - zrow) * 32);
        noc_async_read(zeros, pt + 1024, 512);
        noc_async_read(zeros, pt + 1536, 512);
    }
    // constants: taps (packed per head), selectors, the norm weight tiles c0.., dt_bias / A_neg; z; the a / b rows
    CircularBuffer cw(cb_w), csel(cb_sel), conw(cb_onw), cdt(cb_dtb), can(cb_aneg), cz(cb_z), scr(cb_scr);
    cw.reserve_back(4);
    csel.reserve_back(12);
    conw.reserve_back(NC);
    cdt.reserve_back(1);
    can.reserve_back(1);
    cz.reserve_back(NC * ns);
    scr.reserve_back(1);
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
    }
    for (uint32_t t = 0; t < NC; t++) {
        noc.async_read(onw, conw, tb, {.page_id = c0 + t}, {.offset_bytes = t * tb});
    }
    for (uint32_t t = 0; t < 12; t++) {
        noc.async_read(sel, csel, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
    noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
    const uint32_t pz = cz.get_write_ptr(), ps = scr.get_write_ptr();
    const uint32_t ca = ACOL + hv, cbb = BCOL + hv;
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t r = r0 + s;
        for (uint32_t k = 0; k < NC; k++) {  // z tile: row 0 = row r of qkvzab tile ZT0 + 4hv + c0 + k, the rest 0
            const uint32_t pt = pz + (s * NC + k) * tb;
            noc_async_read(zeros, pt + 32, 480);
            noc_async_read(zeros, pt + 544, 480);
            noc_async_read(zeros, pt + 1024, 512);
            noc_async_read(zeros, pt + 1536, 512);
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xq.get_noc_addr(ZT0 + 4 * hv + c0 + k, f * 512 + r * 32), pt + f * 512, 32);
            }
        }
        noc_async_read(xq.get_noc_addr(ca / 32, ((ca % 32) / 16) * 512 + r * 32), ps + s * 128, 32);
        noc_async_read(xq.get_noc_addr(cbb / 32, ((cbb % 32) / 16) * 512 + r * 32), ps + s * 128 + 64, 32);
    }
    noc.async_read_barrier();
    RTS(1);
    if (!owner) {  // the shared q / k rows are read: the owner may overwrite them
        noc_semaphore_inc(get_noc_addr(owner_x, owner_y, get_semaphore(SEM)), 1);
    }
    if (ns == 1) {  // row r0 + 1 is not an active sequence: its new x row is 0
        for (uint32_t k = 0; k < 24; k++) {
            if (!mine(k / 2)) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* q32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sb + k * 64 + 32);
            for (uint32_t w = 0; w < 8; w++) {
                q32[w] = 0;
            }
        }
    }
    for (uint32_t s = 0; s < ns; s++) {
        for (uint32_t m = 0; m < 4; m++) {
            const uint32_t pt = m == 0 ? px + s * tb : pq + (s * 3 + m - 1) * tb;
            for (uint32_t t = 0; t < 12; t++) {
                if (!mine(t)) {
                    continue;
                }
                for (uint32_t f = 0; f < 2; f++) {
                    uint32_t src = sb + (m * 24 + 2 * t + f) * 64 + s * 32;
                    if constexpr (CMODE == 2) {
                        src = m == 0 ? sb + (2 * t + f) * 64
                                     : sb + ((m == 3 ? 2 : 1) * 24 + 2 * t + f) * 64 + (m == 2 ? 32 : 0);
                    }
                    noc_async_read(get_noc_addr(src), pt + f * 512 + t * 32, 32);
                }
            }
        }
    }
    // a / b as full tiles (every element the scalar): the value in the first 64 bytes, then doubling local copies
    CircularBuffer cav(cb_a), cbv(cb_b);
    cav.reserve_back(ns);
    cbv.reserve_back(ns);
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
    noc.async_read_barrier();  // the packed conv tiles
    cx.push_back(ns);
    cq.push_back(3 * ns);
    cw.push_back(4);
    csel.push_back(12);
    conw.push_back(NC);
    cdt.push_back(1);
    can.push_back(1);
    cz.push_back(NC * ns);
    RTS(2);
    for (uint32_t sz = 64; sz < tb; sz *= 2) {
        for (uint32_t s = 0; s < ns; s++) {
            const uint32_t ta = cav.get_write_ptr() + s * tb, tbb = cbv.get_write_ptr() + s * tb;
            noc_async_read(get_noc_addr(ta), ta + sz, sz);
            noc_async_read(get_noc_addr(tbb), tbb + sz, sz);
        }
        noc.async_read_barrier();
    }
    cav.push_back(ns);
    cbv.push_back(ns);
    csm.push_back(4 * NC);
    csd.push_back(4 * NC);
    RTS(3);

    // ---- the conv-state shift: st0 <- st1 (CMODE 1), st1 <- st2, st2 <- st3, st3 <- x; own v rows first (no wait)
    if constexpr (CMODE == 2) {
        for (uint32_t t = 0; t < 12; t++) {
            if (!mine(t)) {
                continue;
            }
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t sl = (2 * t + f) * 64;
                noc_async_read(get_noc_addr(sb + 24 * 64 + sl + 32), sb + 72 * 64 + sl, 32);
                noc_async_read(get_noc_addr(sb + 48 * 64 + sl), sb + 72 * 64 + sl + 32, 32);
            }
        }
        noc.async_read_barrier();
    }
    auto shift = [&](uint32_t t) {
        const uint32_t c = chan(t);
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512 + roff, sl = (2 * t + f) * 64;
            if constexpr (CMODE == 2) {
                noc_async_write(sb + 72 * 64 + sl, st1.get_noc_addr(c, f * 512), 64);
                noc_async_write(sb + sl, st1.get_noc_addr(c, f * 512 + 64), 64);
            } else {
                if constexpr (CMODE == 1) {
                    noc_async_write(sb + 24 * 64 + sl, st0.get_noc_addr(c, off), 64);
                }
                noc_async_write(sb + 48 * 64 + sl, st1.get_noc_addr(c, off), 64);
                noc_async_write(sb + 72 * 64 + sl, st2.get_noc_addr(c, off), 64);
                noc_async_write(sb + sl, st3.get_noc_addr(c, off), 64);
            }
        }
    };
    for (uint32_t t = 8 + c0; t < 8 + c0 + NC; t++) {
        shift(t);
    }
    if (owner) {
        if constexpr (RF * CS > 1) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
            noc_semaphore_wait(sem, RF * CS - 1);
            noc_semaphore_set(sem, 0);
        }
        for (uint32_t t = 0; t < 8; t++) {
            shift(t);
        }
    }
    RTS(4);

    // ---- the state of the second sequence of the pair (once the compute released the first)
    for (uint32_t s = 1; s < ns; s++) {
        csd.reserve_back(4 * NC);
        csm.reserve_back(4 * NC);
        read_state(r0 + s);
        noc.async_read_barrier();
        csm.push_back(4 * NC);
        csd.push_back(4 * NC);
    }
    noc_async_write_barrier();
    RTS(5);
}
