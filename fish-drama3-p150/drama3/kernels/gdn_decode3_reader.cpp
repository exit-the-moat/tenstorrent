// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode v3 (packed conv, one core per VALUE head hv, 32 cores), reader. Derived
// from v2 (one core per key head). Value heads 2hk and 2hk+1 share the q / k conv channels of key head hk: both cores
// compute them, only the even core writes their new partial sums, after the odd core signals (semaphore) that it has
// read the old ones. The 12 conv channel tiles of the core (q 4hk.., k 64+4hk.., v 128+4hv..) carry their data in row 0 only;
// v2 packs those rows into ONE tile per tensor (row t = channel tile t), so the conv is 1 tile instead of 16 and only
// row 0 of each source tile is moved: the in-projection row x and the three partial-sum rows Q0..Q2 are read as
// 64-byte chunks (rows 0-1 of faces 0 and 1) into aligned staging slots, then NOC-copied into the packed tiles (rows
// 12..31 zero, from MEM_ZEROS). The conv taps come pre-packed per core (4 tiles), the first 12 row-selector tiles
// (E_t: 1 at [0, t]) for the unpack in the compute kernel from a constant tensor.
// Then 4 gate tiles, the scalars a, b (at [0,0]), dt_bias, A_neg and the state of value head hv. Three read batches,
// each pushed when its barrier returns: the conv inputs, then what the conv does not need, then the state.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_gate = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_s = 8,
                   cb_onw = 9, cb_scratch = 10, cb_sel = 11, cb_stage = 12;

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

    // ---- row 0 of each of the 12 channel tiles of x, Q0, Q1, Q2 as 64-byte chunks (rows 0-1 of faces 0 and 1) into
    // 64-byte aligned staging slots (tensor m, tile t, face f -> slot m * 32 + 2t + f), then NOC copies inside L1 put
    // the 32-byte row 0 of each slot at row t of the packed tile; the rest of the packed tiles comes from MEM_ZEROS
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
            const uint32_t off = f * 512, slot = (2 * t + f) * 64;
            noc_async_read(mega.get_noc_addr(c, off), s0 + slot, 64);
            noc_async_read(q0.get_noc_addr(c, off), s0 + tb + slot, 64);
            noc_async_read(q1.get_noc_addr(c, off), s0 + 2 * tb + slot, 64);
            noc_async_read(q2.get_noc_addr(c, off), s0 + 3 * tb + slot, 64);
        }
    }
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t m = 0; m < 4; m++) {  // the non-data parts: rows 12..15 of faces 0 / 1, faces 2 / 3
        const uint32_t p = m == 0 ? px : pq + (m - 1) * tb;
        noc_async_read(zeros, p + 12 * 32, 128);
        noc_async_read(zeros, p + 512 + 12 * 32, 128);
        noc_async_read(zeros, p + 1024, 512);
        noc_async_read(zeros, p + 1536, 512);
    }
    noc.async_read_barrier();
    if (hv & 1) {  // the odd core has read the shared q / k partial sums: the even core may now overwrite them
        noc_semaphore_inc(get_noc_addr(partner_x, partner_y, get_semaphore(SEM)), 1);
    }
    for (uint32_t m = 0; m < 4; m++) {
        const uint32_t p = m == 0 ? px : pq + (m - 1) * tb;
        for (uint32_t t = 0; t < 12; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(get_noc_addr(s0 + m * tb + (2 * t + f) * 64), p + f * 512 + t * 32, 32);
            }
        }
    }
    noc.async_read_barrier();
    cx.push_back(1);
    cq.push_back(3);
    cw.push_back(4);

    // second batch: the row selectors (needed after the conv), the o-norm weight, the gate tiles and the a / b tiles
    // of all 32 heads (mega tiles 256, 257, into the scratch CB, never pushed)
    CircularBuffer cg(cb_gate), scr(cb_scratch);
    cg.reserve_back(4);
    scr.reserve_back(2);
    for (uint32_t t = 0; t < 12; t++) {
        noc.async_read(sel, csel, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(onw, co, tb, {.page_id = t}, {.offset_bytes = t * tb});
        noc.async_read(mega, cg, tb, {.page_id = 258 + 4 * hv + t}, {.offset_bytes = t * tb});
    }
    noc.async_read(mega, scr, tb, {.page_id = 256}, {.offset_bytes = 0});
    noc.async_read(mega, scr, tb, {.page_id = 257}, {.offset_bytes = tb});
    noc.async_read_barrier();
    csel.push_back(12);
    co.push_back(4);
    cg.push_back(4);
    volatile tt_l1_ptr uint16_t* ab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());

    // one bf16 scalar at [0,0] of an otherwise zero tile
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
    // element (0, col) of a 32x32 bf16 tile: face 0 holds columns 0..15, face 1 (offset 256) columns 16..31
    auto row0 = [](uint32_t col) { return col < 16 ? col : 256 + (col - 16); };

    // third batch: dt_bias, A_neg and the state of value head hv; the a / b scalar tiles are built meanwhile
    CircularBuffer cdt(cb_dtb), can(cb_aneg), cs(cb_s);
    cdt.reserve_back(1);
    can.reserve_back(1);
    cs.reserve_back(16);
    noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
    noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
    for (uint32_t t = 0; t < 16; t++) {
        noc.async_read(st, cs, tb, {.page_id = hv * 16 + t}, {.offset_bytes = t * tb});
    }
    put_scalar(cb_a, ab[row0(hv)]);
    put_scalar(cb_b, ab[1024 + row0(hv)]);
    noc.async_read_barrier();
    cdt.push_back(1);
    can.push_back(1);
    cs.push_back(16);
}
