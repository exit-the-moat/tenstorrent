// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 (Exit the Moat, Episode 2): fused one-token GDN decode for the TP qwen36 path (QWEN36_GDN_FUSED_DECODE=1),
// reader. Derived from the Episode 0 Drama 3 kernel (drama3/kernels/gdn_decode3_*), adapted to Qwen3.8-27B per device
// (NK key heads, NV value heads of 128, NV / NK value heads per key head), the 4-deep conv shift register of
// TPGatedDeltaNet (conv_states[0..3], bf16 [1, Bmax, qkv_dim]), the fp32 recurrent state [Bmax, NV, 128, 128] and up
// to 8 sequences (rows of the qkvzab tile).
//
// One core per (value head hv, row pair p): sequences r0 = 2p and r0 + 1 (the second only if r0 + 1 < B). A row pair
// is one 64-byte chunk per face row, so this core owns the conv-state chunks it rewrites (no 32-byte DRAM writes).
// The q / k conv channels of key head hk = hv / RF are shared by the RF value-head cores of that key head: all read
// the old rows, only the owner (hv % RF == 0) writes the shifted rows, after the others signal (semaphore) that their
// reads are done.
//
// Packed conv tiles (one per tensor and sequence): row t = row r of channel tile t (t < 4: q, < 8: k, < 12: v).

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
    constexpr uint32_t ZT0 = get_compile_time_arg_val(3);   // first z tile of the in-projection row
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);  // column of a for head 0
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);  // column of b for head 0
    // conv history: 0 = three tensors st1..st3 (rows r of each), 1 = the TP 4-deep shift register st0..st3 (the oldest
    // row goes to st0), 2 = ONE tensor st1 with the history in rows 0..2 (single-device fused conv state, B = 1)
    constexpr uint32_t CMODE = get_compile_time_arg_val(6);
    constexpr uint32_t STF32 = get_compile_time_arg_val(7);  // 1: fp32 recurrent state, 0: bf16
    constexpr uint32_t RF = NV / NK;
    constexpr auto xa = TensorAccessorArgs<8>();
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
    // the in-projection row starts with [q NK*128 | k NK*128 | v NV*128]; z (4 tiles per head), a, b are placed by
    // ZT0 / ACOL / BCOL (TP qkvzab: [qkv | z | a | b]; single-device mega: [qkv | a | b | z])
    constexpr uint32_t KT0 = 4 * NK, VT0 = 8 * NK;

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t p = get_arg_val<uint32_t>(i++);
    const uint32_t B = get_arg_val<uint32_t>(i++);
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
    const uint32_t owner_x = get_arg_val<uint32_t>(i++);  // NOC coordinates of the owner core of this key head / pair
    const uint32_t owner_y = get_arg_val<uint32_t>(i++);

    const uint32_t hk = hv / RF;
    const bool owner = (hv % RF) == 0;
    const uint32_t r0 = 2 * p;
    const uint32_t ns = (r0 + 1 < B) ? 2 : 1;
    const uint32_t roff = r0 * 32;  // byte offset of row r0 inside a face (rows r0, r0 + 1 = one 64-byte chunk)

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

    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
#ifdef GDNQ_TS
    // ablation: per-core stage timestamps (wall clock, low 32 bits) in one shared tile of CB 27: reader 0..15,
    // compute 16..47, writer 48..63; the writer stores the tile
    CircularBuffer cts(27);
    cts.reserve_back(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_write_ptr());
    for (uint32_t k = 1; k < 64; k++) {
        tsp[k] = 0;
    }
#define RTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    RTS(0);
    cts.push_back(1);
#else
#define RTS(k)
#endif

    // ---- batch 1: the conv inputs. Chunk (m, t, f) -> staging slot s0 + (m * 24 + 2t + f) * 64; m 0 = x (qkvzab, L1),
    // 1..3 = conv_states[1..3] (DRAM, 64-byte reads into 64-byte aligned slots)
    CircularBuffer stg(cb_stage), cx(cb_x), cq(cb_q);
    stg.reserve_back(3);
    cx.reserve_back(ns);
    cq.reserve_back(3 * ns);
    const uint32_t sb = stg.get_write_ptr(), px = cx.get_write_ptr(), pq = cq.get_write_ptr();
    for (uint32_t t = 0; t < 12; t++) {
        const uint32_t c = chan(t);
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512 + roff, sl = (2 * t + f) * 64;
            noc_async_read(xq.get_noc_addr(c, off), sb + sl, 64);
            if constexpr (CMODE == 2) {  // rows 0-1 and 2-3 of the one history tensor
                noc_async_read(st1.get_noc_addr(c, f * 512), sb + 24 * 64 + sl, 64);
                noc_async_read(st1.get_noc_addr(c, f * 512 + 64), sb + 48 * 64 + sl, 64);
            } else {
                noc_async_read(st1.get_noc_addr(c, off), sb + 24 * 64 + sl, 64);
                noc_async_read(st2.get_noc_addr(c, off), sb + 48 * 64 + sl, 64);
                noc_async_read(st3.get_noc_addr(c, off), sb + 72 * 64 + sl, 64);
            }
        }
    }
    // packed tiles: rows 12..15 of faces 0 / 1 and faces 2 / 3 are zero
    for (uint32_t k = 0; k < 4 * ns; k++) {
        const uint32_t pt = k < ns ? px + k * tb : pq + (k - ns) * tb;
        noc_async_read(zeros, pt + 12 * 32, 128);
        noc_async_read(zeros, pt + 512 + 12 * 32, 128);
        noc_async_read(zeros, pt + 1024, 512);
        noc_async_read(zeros, pt + 1536, 512);
    }
    noc.async_read_barrier();
    RTS(1);
    if (!owner) {  // the shared q / k rows are read: the owner may overwrite them
        noc_semaphore_inc(get_noc_addr(owner_x, owner_y, get_semaphore(SEM)), 1);
    }
    if (ns == 1) {  // row r0 + 1 is not an active sequence: its new x row is 0 (as the padded reference shift)
        for (uint32_t k = 0; k < 24; k++) {
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
                for (uint32_t f = 0; f < 2; f++) {
                    uint32_t src = sb + (m * 24 + 2 * t + f) * 64 + s * 32;
                    if constexpr (CMODE == 2) {  // h0 = row 0, h1 = row 1 of chunk A (m 1), h2 = row 0 of chunk B (m 2)
                        src = m == 0 ? sb + (2 * t + f) * 64
                                     : sb + ((m == 3 ? 2 : 1) * 24 + 2 * t + f) * 64 + (m == 2 ? 32 : 0);
                    }
                    noc_async_read(get_noc_addr(src), pt + f * 512 + t * 32, 32);
                }
            }
        }
    }
    noc.async_read_barrier();
    cx.push_back(ns);
    cq.push_back(3 * ns);
    RTS(2);

    // ---- the conv-state shift: st0 <- st1 (CMODE 1 only), st1 <- st2, st2 <- st3, st3 <- x (both rows of the pair)
    if (owner) {
        if constexpr (RF > 1) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
            noc_semaphore_wait(sem, RF - 1);
            noc_semaphore_set(sem, 0);
        }
    }
    RTS(3);
    if constexpr (CMODE == 2) {
        // new rows 0-1 = [h1, h2] assembled in slot region 3; new rows 2-3 = [x, 0] = the x chunk as it is (row 3 is
        // tile padding)
        for (uint32_t t = owner ? 0 : 8; t < 12; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t sl = (2 * t + f) * 64;
                noc_async_read(get_noc_addr(sb + 24 * 64 + sl + 32), sb + 72 * 64 + sl, 32);
                noc_async_read(get_noc_addr(sb + 48 * 64 + sl), sb + 72 * 64 + sl + 32, 32);
            }
        }
        noc.async_read_barrier();
    }
    for (uint32_t t = owner ? 0 : 8; t < 12; t++) {
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
    }

    // ---- batch 2: taps, selectors, norm weight, dt_bias / A_neg (full tiles), z rows, the a / b rows
    CircularBuffer cw(cb_w), csel(cb_sel), conw(cb_onw), cdt(cb_dtb), can(cb_aneg), cz(cb_z), scr(cb_scr);
    cw.reserve_back(4);
    csel.reserve_back(12);
    conw.reserve_back(4);
    cdt.reserve_back(1);
    can.reserve_back(1);
    cz.reserve_back(4 * ns);
    scr.reserve_back(1);
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
        noc.async_read(onw, conw, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    for (uint32_t t = 0; t < 12; t++) {
        noc.async_read(sel, csel, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
    noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
    const uint32_t pz = cz.get_write_ptr(), ps = scr.get_write_ptr();
    const uint32_t ca = ACOL + hv, cbb = BCOL + hv;  // columns of a and b
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t r = r0 + s;
        for (uint32_t k = 0; k < 4; k++) {  // z tile: row 0 = row r of qkvzab tile ZT0 + 4hv + k, the rest 0
            const uint32_t pt = pz + (s * 4 + k) * tb;
            noc_async_read(zeros, pt + 32, 480);
            noc_async_read(zeros, pt + 544, 480);
            noc_async_read(zeros, pt + 1024, 512);
            noc_async_read(zeros, pt + 1536, 512);
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xq.get_noc_addr(ZT0 + 4 * hv + k, f * 512 + r * 32), pt + f * 512, 32);
            }
        }
        // the 16-value face row holding a (and b) of row r, into 64-byte aligned scratch slots
        noc_async_read(xq.get_noc_addr(ca / 32, ((ca % 32) / 16) * 512 + r * 32), ps + s * 128, 32);
        noc_async_read(xq.get_noc_addr(cbb / 32, ((cbb % 32) / 16) * 512 + r * 32), ps + s * 128 + 64, 32);
    }
    noc.async_read_barrier();
    RTS(4);
    cw.push_back(4);
    csel.push_back(12);
    conw.push_back(4);
    cdt.push_back(1);
    can.push_back(1);
    cz.push_back(4 * ns);

    // a / b as full tiles (every element the scalar)
    CircularBuffer cav(cb_a), cbv(cb_b);
    cav.reserve_back(ns);
    cbv.reserve_back(ns);
    for (uint32_t s = 0; s < ns; s++) {
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps + s * 128);
        const uint32_t va = row[ca % 16], vb = row[32 + cbb % 16];
        volatile tt_l1_ptr uint32_t* ta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cav.get_write_ptr() + s * tb);
        volatile tt_l1_ptr uint32_t* tbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbv.get_write_ptr() + s * tb);
        const uint32_t wa = va | (va << 16), wb = vb | (vb << 16);
        for (uint32_t w = 0; w < tb / 4; w++) {
            ta[w] = wa;
            tbp[w] = wb;
        }
    }
    cav.push_back(ns);
    cbv.push_back(ns);
    RTS(5);

    // ---- batch 3 (per sequence): the fp32 state of (row r, head hv), once for the SFPU (cb_sdst, unpack to dest)
    // and once for the matmuls (cb_smm, local copy)
    CircularBuffer csd(cb_sdst), csm(cb_smm);
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t r = r0 + s;
        csd.reserve_back(16);
        csm.reserve_back(16);
        const uint32_t pd = csd.get_write_ptr(), pm = csm.get_write_ptr();
#ifndef GDNQ_NOSTATE
        for (uint32_t t = 0; t < 16; t++) {
            noc_async_read(rec.get_noc_addr((r * NV + hv) * 16 + t), pm + t * tf, tf);
        }
        noc.async_read_barrier();
#endif
        csm.push_back(16);
        RTS(6 + 2 * s);
#ifndef GDNQ_NOSTATE
        noc_async_read(get_noc_addr(pm), pd, 16 * tf);
        noc.async_read_barrier();
#endif
        csd.push_back(16);
        RTS(7 + 2 * s);
    }
    noc_async_write_barrier();
    RTS(10);
}
