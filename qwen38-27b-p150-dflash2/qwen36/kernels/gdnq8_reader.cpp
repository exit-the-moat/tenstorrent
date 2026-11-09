// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn (Exit the Moat, Episode 2): fused one-token GDN decode v8 (QWEN36_GDN_FAST=8), reader. v6 without the first
// state read: the writer RISC (idle at the start) reads the state of the first NST sequences right after the gate
// inputs (gdnq8_writer.cpp), in parallel with the conv rows here; this RISC reads the remaining sequence (NC 4 / 2
// with 2 sequences) once the writer pushed the first and the compute made room.
// v6: v4 (below) plus, with
// more than one row pair (B > 2): the conv rows of ALL pairs are read once per channel tile by the pair-0 core of the
// (head, column slice) as one RW * 32-byte chunk per face (RW = 2 * pairs rows, 4x fewer DRAM transactions at B = 8;
// staging CB 34), the cores of the other pairs read their 64-byte pieces from its L1 after semaphore 2, and the pair-0
// core alone writes the shifted conv state back, again as whole chunks.
// v4: v3 (below) plus:
//   - the shared q / k conv rows are read from DRAM by the owner core of the key head only (hv % RF == 0, ci == 0)
//     and forwarded L1 -> L1 to the other RF * CS - 1 cores (faces 0 / 1 = the first 1 KB of each of the 4 packed
//     tiles, the same L1 address on every core), then semaphore 0 there; the owner rewrites them in DRAM without
//     waiting for anyone (nobody else reads them);
//   - a / b / dt_bias / A_neg come from the writer RISC (gdnq4_writer.cpp), so the compute can do the gates first.
// v3:  The data of
// gdnq_reader.cpp (TP path, CMODE 1 only), laid out so the DRAM reads land where the compute needs them:
//   - packed conv tiles hold BOTH sequences of the row pair: channel tile t, sequence s -> packed row
//     R(t, s) = 2 (t % 8) + s of faces f + 2 (t / 8). A 64-byte chunk (rows r0, r0 + 1 of face f of a conv tensor tile)
//     is read straight to its packed position (64-byte aligned): no staging, no local copies, one conv for both
//     sequences. The conv-state shift writes go out from the packed tiles (st0 <- st1, st1 <- st2, st2 <- st3,
//     st3 <- x, with x row r0 + 1 zeroed when it is not an active sequence).
//   - the selector tiles E_R (row 0 picks packed row R) are generated in L1 (zeros + one 1.0), no DRAM reads;
//   - reads are phased by first use: A = conv rows + taps (+ selectors), B = gates / z / norm weight / state.
// CS cores per (value head, row pair) as in gdnq2_reader.cpp (core ci owns state column tiles [c0, c0 + NC)).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_sdst = 8,
                   cb_onw = 9, cb_scr = 10, cb_sel = 11, cb_smm = 13;

void kernel_main() {
    constexpr uint32_t SEM = get_compile_time_arg_val(0);
    constexpr uint32_t NK = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t ZT0 = get_compile_time_arg_val(3);
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);
    constexpr uint32_t CMODE = get_compile_time_arg_val(6);
    static_assert(CMODE == 1, "v3: the TP 4-deep conv shift register only");
    constexpr uint32_t STF32 = get_compile_time_arg_val(7);
    constexpr uint32_t CS = get_compile_time_arg_val(8);
    constexpr uint32_t NC = 4 / CS;
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
    i++;  // the selector tensor (v1 / v2): generated here instead
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);
    i += 2;  // (v3: owner coordinates)
    const uint32_t nfw = get_arg_val<uint32_t>(i++);  // q / k forward targets (owner cores), NOC x / y pairs
    const uint32_t ifw = i;
    i += 2 * nfw;
    const uint32_t ncons = get_arg_val<uint32_t>(i++);  // merge, pair-0 cores: the other pairs' cores of this slice
    const uint32_t icons = i;
    i += 2 * ncons;
    const uint32_t src_x = get_arg_val<uint32_t>(i++), src_y = get_arg_val<uint32_t>(i++);  // merge: the pair-0 core

    const uint32_t hk = hv / RF;
    const uint32_t c0 = ci * NC;
    const bool owner = (hv % RF) == 0 && ci == 0;
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;
    const uint32_t roff = r0 * 32;
    const uint32_t np = (B + 1) / 2;
    const bool merge = np > 1;
    const uint32_t RW = 2 * np;  // conv rows per chunk in merge mode

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048;
    const auto xq = TensorAccessor(xa, x_addr, tb);
    const auto st0 = TensorAccessor(s0a, s0_addr, tb);
    const auto st1 = TensorAccessor(s1a, s1_addr, tb);
    const auto st2 = TensorAccessor(s2a, s2_addr, tb);
    const auto st3 = TensorAccessor(s3a, s3_addr, tb);
    const auto wp = TensorAccessor(wpa, wp_addr, tb);
    const auto rec = TensorAccessor(reca, rec_addr, tf);
    const auto onw = TensorAccessor(onwa, onw_addr, tb);

    auto chan = [&](uint32_t t) -> uint32_t {
        return t < 4 ? 4 * hk + t : (t < 8 ? KT0 + 4 * hk + (t - 4) : VT0 + 4 * hv + (t - 8));
    };
    // byte offset of chunk (t, f) (packed rows R(t, 0), R(t, 1)) inside a packed tile
    auto poff = [&](uint32_t t, uint32_t f) -> uint32_t { return (f + 2 * (t / 8)) * 512 + (t % 8) * 64; };

    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
#ifdef GDNQ_TS
    CircularBuffer cts(27);
    cts.reserve_back(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_write_ptr());
#define RTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    RTS(0);
    cts.push_back(1);
#else
#define RTS(k)
#endif

    // ---- phase A: conv rows (packed), taps, selectors
    CircularBuffer cx(cb_x), cq(cb_q), cw(cb_w), csel(cb_sel);
    cx.reserve_back(1);
    cq.reserve_back(3);
    cw.reserve_back(4);
    constexpr uint32_t NSEL = 8 + NC;  // selectors per sequence: q 0..3, k 4..7, own v
    csel.reserve_back(NSEL * ns);
    const uint32_t px = cx.get_write_ptr(), pq = cq.get_write_ptr(), psel = csel.get_write_ptr();
    // merge staging: chunk (m = x / st1 / st2 / st3, t, f) at ((m * 12 + t) * 2 + f) * 256, rows 0 .. RW - 1
    CircularBuffer cstg(34);
    cstg.reserve_back(12);
    const uint32_t ps34 = cstg.get_write_ptr();
    auto soff = [&](uint32_t m, uint32_t t, uint32_t f) -> uint32_t { return ((m * 12 + t) * 2 + f) * 256; };
    auto own_t = [&](uint32_t t) -> bool { return t < 8 ? owner : (t - 8 >= c0 && t - 8 < c0 + NC); };
    if (!merge) {
        for (uint32_t t = 0; t < 12; t++) {
            if (!own_t(t)) {
                continue;
            }
            const uint32_t c = chan(t);
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t off = f * 512 + roff, d = poff(t, f);
                noc_async_read(xq.get_noc_addr(c, off), px + d, 64);
                noc_async_read(st1.get_noc_addr(c, off), pq + d, 64);
                noc_async_read(st2.get_noc_addr(c, off), pq + tb + d, 64);
                noc_async_read(st3.get_noc_addr(c, off), pq + 2 * tb + d, 64);
            }
        }
    } else if (p == 0) {
        for (uint32_t t = 0; t < 12; t++) {
            if (!own_t(t)) {
                continue;
            }
            const uint32_t c = chan(t);
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xq.get_noc_addr(c, f * 512), ps34 + soff(0, t, f), RW * 32);
                noc_async_read(st1.get_noc_addr(c, f * 512), ps34 + soff(1, t, f), RW * 32);
                noc_async_read(st2.get_noc_addr(c, f * 512), ps34 + soff(2, t, f), RW * 32);
                noc_async_read(st3.get_noc_addr(c, f * 512), ps34 + soff(3, t, f), RW * 32);
            }
        }
    }
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
    }
    // zero rows of the packed tiles (disjoint from the chunks read above): faces 2 / 3 rows 8..15 (no channel tile
    // t >= 12) and the v rows of the other cores' columns
    for (uint32_t k = 0; k < 4; k++) {
        const uint32_t pt = k == 0 ? px : pq + (k - 1) * tb;
        for (uint32_t f = 2; f < 4; f++) {
            noc_async_read(zeros, pt + f * 512 + 256, 256);
            for (uint32_t v = 0; v < 4; v++) {
                if (v < c0 || v >= c0 + NC) {
                    noc_async_read(zeros, pt + f * 512 + v * 64, 64);
                }
            }
        }
    }
    for (uint32_t k = 0; k < NSEL * ns; k++) {
        for (uint32_t q4 = 0; q4 < 4; q4++) {
            noc_async_read(zeros, psel + k * tb + q4 * 512, 512);
        }
    }
    noc.async_read_barrier();
    RTS(1);
    if (merge) {
        volatile tt_l1_ptr uint32_t* semv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        uint64_t sbase;
        if (p == 0) {
            if (B & 1) {  // row B (the last pair's second row) is not an active sequence: its new conv row is 0
                for (uint32_t t = 0; t < 12; t++) {
                    if (!own_t(t)) {
                        continue;
                    }
                    for (uint32_t f = 0; f < 2; f++) {
                        volatile tt_l1_ptr uint32_t* q32 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps34 + soff(0, t, f) + B * 32);
                        for (uint32_t w = 0; w < 8; w++) {
                            q32[w] = 0;
                        }
                    }
                }
            }
            for (uint32_t k = 0; k < ncons; k++) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(icons + 2 * k), get_arg_val<uint32_t>(icons + 2 * k + 1),
                                 get_semaphore(2)),
                    1);
            }
            sbase = get_noc_addr(ps34);
        } else {
            noc_semaphore_wait(semv, 1);
            sbase = get_noc_addr(src_x, src_y, ps34);
        }
        // this pair's 64-byte pieces (rows 2p, 2p + 1) into the packed tiles
        for (uint32_t t = 0; t < 12; t++) {
            if (!own_t(t)) {
                continue;
            }
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t d = poff(t, f);
                noc_async_read(sbase + soff(0, t, f) + p * 64, px + d, 64);
                noc_async_read(sbase + soff(1, t, f) + p * 64, pq + d, 64);
                noc_async_read(sbase + soff(2, t, f) + p * 64, pq + tb + d, 64);
                noc_async_read(sbase + soff(3, t, f) + p * 64, pq + 2 * tb + d, 64);
            }
        }
        noc.async_read_barrier();
    }
    // selector E_R: 1.0 (bf16 0x3f80) at row 0, column R
    for (uint32_t s = 0; s < ns; s++) {
        for (uint32_t k = 0; k < NSEL; k++) {
            const uint32_t t = k < 8 ? k : 8 + c0 + (k - 8);
            const uint32_t R = (t / 8) * 16 + 2 * (t % 8) + s;
            volatile tt_l1_ptr uint16_t* e =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(psel + (s * NSEL + k) * tb + (R / 16) * 512);
            e[R % 16] = 0x3f80;
        }
    }
    if (ns == 1 && !merge) {  // x row r0 + 1 is not an active sequence: its new conv row is 0
        for (uint32_t t = owner ? 0 : 8; t < 12; t++) {
            if (t >= 8 && (t - 8 < c0 || t - 8 >= c0 + NC)) {
                continue;
            }
            for (uint32_t f = 0; f < 2; f++) {
                volatile tt_l1_ptr uint32_t* q32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(px + poff(t, f) + 32);
                for (uint32_t w = 0; w < 8; w++) {
                    q32[w] = 0;
                }
            }
        }
    }
    if constexpr (RF * CS > 1) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
        if (owner) {  // forward the q / k rows (faces 0 / 1 of the 4 packed tiles) to the other cores of the key head
            for (uint32_t k = 0; k < 2 * nfw; k += 2) {
                const uint32_t nx = get_arg_val<uint32_t>(ifw + k), ny = get_arg_val<uint32_t>(ifw + k + 1);
                noc_async_write(px, get_noc_addr(nx, ny, px), 1024);
                for (uint32_t m = 0; m < 3; m++) {
                    noc_async_write(pq + m * tb, get_noc_addr(nx, ny, pq + m * tb), 1024);
                }
            }
            noc_async_write_barrier();
            for (uint32_t k = 0; k < 2 * nfw; k += 2) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(ifw + k), get_arg_val<uint32_t>(ifw + k + 1), get_semaphore(SEM)), 1);
            }
        } else {
            noc_semaphore_wait(sem, 1);
        }
    }
    cx.push_back(1);
    cq.push_back(3);
    cw.push_back(4);
    csel.push_back(NSEL * ns);
    RTS(2);

    // ---- phase B: the state (both sequences if the CB holds them), gates, z, norm weight
    CircularBuffer csd(cb_sdst), csm(cb_smm), conw(cb_onw), cz(cb_z);
    const uint32_t nst = (8 * NC * ns <= 16) ? ns : 1;  // sequences whose state the writer reads (same rule there)
    conw.reserve_back(NC);
    cz.reserve_back(NC * ns);
    const uint32_t pz = cz.get_write_ptr();
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
    }
    for (uint32_t t = 0; t < NC; t++) {
        noc.async_read(onw, conw, tb, {.page_id = c0 + t}, {.offset_bytes = t * tb});
    }
    noc.async_read_barrier();
    cz.push_back(NC * ns);
    conw.push_back(NC);
    RTS(3);

    // ---- the conv-state shift from the packed tiles: own v rows first (no wait), then (owner) the shared q / k rows
    auto shift = [&](uint32_t t) {
        const uint32_t c = chan(t);
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512 + roff, d = poff(t, f);
            noc_async_write(pq + d, st0.get_noc_addr(c, off), 64);
            noc_async_write(pq + tb + d, st1.get_noc_addr(c, off), 64);
            noc_async_write(pq + 2 * tb + d, st2.get_noc_addr(c, off), 64);
            noc_async_write(px + d, st3.get_noc_addr(c, off), 64);
        }
    };
    if (!merge) {
        for (uint32_t t = owner ? 0 : 8 + c0; t < 8 + c0 + NC; t++) {
            if (t >= 8 && t < 8 + c0) {
                continue;
            }
            shift(t);
        }
    } else if (p == 0) {  // the shifted rows of all pairs, as whole chunks
        for (uint32_t t = 0; t < 12; t++) {
            if (!own_t(t)) {
                continue;
            }
            const uint32_t c = chan(t);
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(ps34 + soff(1, t, f), st0.get_noc_addr(c, f * 512), RW * 32);
                noc_async_write(ps34 + soff(2, t, f), st1.get_noc_addr(c, f * 512), RW * 32);
                noc_async_write(ps34 + soff(3, t, f), st2.get_noc_addr(c, f * 512), RW * 32);
                noc_async_write(ps34 + soff(0, t, f), st3.get_noc_addr(c, f * 512), RW * 32);
            }
        }
    }
    RTS(4);

    // ---- the state of the second sequence when the CB holds one sequence only (NC 4, 2 sequences)
    for (uint32_t s = nst; s < ns; s++) {
        // wait until the writer pushed the first sequences on both CBs (the shared tiles-received counters; NOT
        // cb_wait_front: it reads tiles-acked once and would spin forever once the compute popped them), then room;
        // the CB slot by hand (this RISC's own write pointer never moved): tile (4 NC s) mod 16
        for (uint32_t cbi : {cb_smm, cb_sdst}) {
            const uintptr_t rp = (uintptr_t)get_cb_tiles_received_ptr(cbi);
            while ((uint16_t)reg_read(rp) < 4 * NC * nst) {
                invalidate_l1_cache();
            }
        }
        csd.reserve_back(4 * NC);
        csm.reserve_back(4 * NC);
        const uint32_t slot = ((4 * NC * s) % 16) * tf;
        const uint32_t pm = csm.get_write_ptr() + slot, pd = csd.get_write_ptr() + slot;
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_read(rec.get_noc_addr(((r0 + s) * NV + hv) * 16 + ki * 4 + c0 + j), pm + (ki * NC + j) * tf, tf);
            }
        }
        noc.async_read_barrier();
        csm.push_back(4 * NC);
        noc_async_read(get_noc_addr(pm), pd, 4 * NC * tf);
        noc.async_read_barrier();
        csd.push_back(4 * NC);
    }
    noc_async_write_barrier();
    RTS(5);
}
