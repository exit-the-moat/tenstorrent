// drama3 port10: v5 = v4 for 16 < B <= 32: the conv partial sums [1, 2B, 8192] span two tile rows (row 2b in tile
// row 2b / 32, pages + 256), and the output padding rows B..31 are zeroed row by row.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: fused GDN decode v4 = v3 (kernels/gdn_decode3_reader.cpp) for B one-token sequences at once, reader.
// Core (hv, g) runs value head hv of the sequences b = g, g + G, ... < B, one after the other (the v3 body per
// sequence). Batch layouts (B <= 8):
//   mega in-projection [1, B, 12352]: sequence b = row b (read as the 64-byte chunk of rows b & ~1, b | 1);
//   conv partial sums Q0..Q2 [1, 2B, 8192]: sequence b = row 2b, row 2b + 1 stays 0, so a 64-byte chunk (two rows)
//     belongs to one sequence and cores of different sequences never write the same chunk;
//   recurrent state [B, 32, 128, 128]: sequence b = pages (b * 32 + hv) * 16 ..;
//   the gate tiles of row b are moved to row 0 (rows 1..31 zero), the a / b scalars read at element (b, hv).
// The odd core of a key head signals the even one once per sequence (semaphore count = sequences read).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_gate = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_s = 8,
                   cb_onw = 9, cb_scratch = 10, cb_sel = 11, cb_stage = 12;

// byte offset of (row r, column 16 f) in a bf16 32x32 tile (faces of 16x16, row-major inside a face)
inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr auto mega_a = TensorAccessorArgs<1>();
    constexpr auto q0_a = TensorAccessorArgs<mega_a.next_compile_time_args_offset()>();
    constexpr auto q1_a = TensorAccessorArgs<q0_a.next_compile_time_args_offset()>();
    constexpr auto q2_a = TensorAccessorArgs<q1_a.next_compile_time_args_offset()>();
    constexpr auto wp_a = TensorAccessorArgs<q2_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<wp_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<sel_a.next_compile_time_args_offset()>();
    constexpr auto dtb_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto aneg_a = TensorAccessorArgs<dtb_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<aneg_a.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t hk = hv / 2;
    const uint32_t g0 = get_arg_val<uint32_t>(i++);
    const uint32_t G = get_arg_val<uint32_t>(i++);
    const uint32_t B = get_arg_val<uint32_t>(i++);
    const uint32_t mega_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t sel_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);
    const uint32_t partner_x = get_arg_val<uint32_t>(i++);  // NOC coordinates of the even core of this key head
    const uint32_t partner_y = get_arg_val<uint32_t>(i++);
    constexpr uint32_t SEM = get_compile_time_arg_val(0);

    const uint32_t tb = get_tile_size(cb_x);  // bf16; every tensor here is bf16 TILE
    const auto mega = TensorAccessor(mega_a, mega_addr, tb);
    const auto q0 = TensorAccessor(q0_a, q0_addr, tb);
    const auto q1 = TensorAccessor(q1_a, q1_addr, tb);
    const auto q2 = TensorAccessor(q2_a, q2_addr, tb);
    const auto wp = TensorAccessor(wp_a, wp_addr, tb);
    const auto sel = TensorAccessor(sel_a, sel_addr, tb);
    const auto st = TensorAccessor(s_a, s_addr, tb);
    const auto dtb = TensorAccessor(dtb_a, dtb_addr, tb);
    const auto aneg = TensorAccessor(aneg_a, aneg_addr, tb);
    const auto onw = TensorAccessor(onw_a, onw_addr, tb);

    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    // element (r, c) of a bf16 tile in uint16 units
    auto elem = [](uint32_t r, uint32_t c) { return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + (c % 16); };

    for (uint32_t b = g0; b < B; b += G) {
        const uint32_t xb = rowoff(b & ~1u, 0) - rowoff(0, 0), xsub = (b & 1) * 32;  // x row b inside its 2-row chunk
        const uint32_t qb = rowoff((2 * b) % 32, 0), qp = ((2 * b) / 32) * 256;                                        // Q row 2b (+ zero row 2b + 1)
        CircularBuffer stg(cb_stage), cw(cb_w), csel(cb_sel), co(cb_onw), cx(cb_x), cq(cb_q);
        stg.reserve_back(4);
        cw.reserve_back(4);
        csel.reserve_back(12);
        co.reserve_back(4);
        cx.reserve_back(1);
        cq.reserve_back(3);
        const uint32_t s0 = stg.get_write_ptr(), px = cx.get_write_ptr(), pq = cq.get_write_ptr();
        for (uint32_t t = 0; t < 12; t++) {
            const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 4 * hv + (t - 8));
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t slot = (2 * t + f) * 64;
                noc_async_read(mega.get_noc_addr(c, xb + f * 512), s0 + slot, 64);
                noc_async_read(q0.get_noc_addr(qp + c, qb + f * 512), s0 + tb + slot, 64);
                noc_async_read(q1.get_noc_addr(qp + c, qb + f * 512), s0 + 2 * tb + slot, 64);
                noc_async_read(q2.get_noc_addr(qp + c, qb + f * 512), s0 + 3 * tb + slot, 64);
            }
        }
        for (uint32_t t = 0; t < 4; t++) {
            noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
        }
        for (uint32_t m = 0; m < 4; m++) {  // the non-data parts: rows 12..15 of faces 0 / 1, faces 2 / 3
            const uint32_t p = m == 0 ? px : pq + (m - 1) * tb;
            noc_async_read(zeros, p + 12 * 32, 128);
            noc_async_read(zeros, p + 512 + 12 * 32, 128);
            noc_async_read(zeros, p + 1024, 512);
            noc_async_read(zeros, p + 1536, 512);
        }
        noc.async_read_barrier();
        if (hv & 1) {  // the odd core has read sequence b's shared q / k partial sums
            noc_semaphore_inc(get_noc_addr(partner_x, partner_y, get_semaphore(SEM)), 1);
        }
        for (uint32_t m = 0; m < 4; m++) {
            const uint32_t p = m == 0 ? px : pq + (m - 1) * tb;
            const uint32_t sub = m == 0 ? xsub : 0;
            for (uint32_t t = 0; t < 12; t++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_read(get_noc_addr(s0 + m * tb + (2 * t + f) * 64 + sub), p + f * 512 + t * 32, 32);
                }
            }
        }
        noc.async_read_barrier();
        cx.push_back(1);
        cq.push_back(3);
        cw.push_back(4);

        // second batch: row selectors, o-norm weight, the gate tiles (row b moved to row 0: 64-byte chunks into the
        // staging tile, then 32-byte copies into zeroed tiles), the a / b tiles of all heads (scratch, never pushed)
        CircularBuffer cg(cb_gate), scr(cb_scratch);
        cg.reserve_back(4);
        scr.reserve_back(2);
        const uint32_t pg = cg.get_write_ptr();
        for (uint32_t t = 0; t < 12; t++) {
            noc.async_read(sel, csel, tb, {.page_id = t}, {.offset_bytes = t * tb});
        }
        for (uint32_t t = 0; t < 4; t++) {
            noc.async_read(onw, co, tb, {.page_id = t}, {.offset_bytes = t * tb});
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(mega.get_noc_addr(258 + 4 * hv + t, xb + f * 512), s0 + (2 * t + f) * 64, 64);
            }
            for (uint32_t k = 0; k < tb; k += 512) {
                noc_async_read(zeros, pg + t * tb + k, 512);
            }
        }
        noc.async_read(mega, scr, tb, {.page_id = 256}, {.offset_bytes = 0});
        noc.async_read(mega, scr, tb, {.page_id = 257}, {.offset_bytes = tb});
        noc.async_read_barrier();
        for (uint32_t t = 0; t < 4; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(get_noc_addr(s0 + (2 * t + f) * 64 + xsub), pg + t * tb + f * 512, 32);
            }
        }
        noc.async_read_barrier();
        csel.push_back(12);
        co.push_back(4);
        cg.push_back(4);
        volatile tt_l1_ptr uint16_t* ab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());

        auto put_scalar = [&](uint32_t cb_id, uint16_t v) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(1);
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
            for (uint32_t k = 0; k < tb / 4; k++) {
                p[k] = 0;
            }
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(p)[0] = v;
            cb.push_back(1);
        };

        // third batch: dt_bias, A_neg and sequence b's state of value head hv
        CircularBuffer cdt(cb_dtb), can(cb_aneg), cs(cb_s);
        cdt.reserve_back(1);
        can.reserve_back(1);
        cs.reserve_back(16);
        noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
        noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
        for (uint32_t t = 0; t < 16; t++) {
            noc.async_read(st, cs, tb, {.page_id = (b * 32 + hv) * 16 + t}, {.offset_bytes = t * tb});
        }
        put_scalar(cb_a, ab[elem(b, hv)]);
        put_scalar(cb_b, ab[1024 + elem(b, hv)]);
        noc.async_read_barrier();
        cdt.push_back(1);
        can.push_back(1);
        cs.push_back(16);
    }
}
