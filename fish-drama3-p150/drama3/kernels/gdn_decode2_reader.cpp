// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode v2 (packed conv), reader. One core per key head hk (value heads 2hk,
// 2hk+1). The 16 conv channel tiles of the core (q 4hk.., k 64+4hk.., v 128+8hk..) carry their data in row 0 only;
// v2 packs those rows into ONE tile per tensor (row t = channel tile t), so the conv is 1 tile instead of 16 and only
// row 0 of each source tile is moved: the in-projection row x and the three partial-sum rows Q0..Q2 are read as
// 64-byte chunks (rows 0-1 of faces 0 and 1; row 1 is the zero padding row) into a staging buffer, then copied into
// the packed tiles (rows 16..31 zero). The conv taps come pre-packed per core (4 tiles), the 16 row-selector tiles
// (E_t: 1 at [0, t]) for the unpack in the compute kernel from a constant tensor.
// Then, as v1: 8 gate tiles, per value head j the scalars a_j, b_j (at [0,0]), dt_bias_j, A_neg_j and the state.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_gate = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_s = 8,
                   cb_onw = 9, cb_scratch = 10, cb_sel = 11, cb_stage = 12;

void kernel_main() {
    constexpr auto mega_a = TensorAccessorArgs<0>();
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
    const uint32_t hk = get_arg_val<uint32_t>(i++);
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
    auto read_n = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = page + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // ---- row chunks of x, Q0, Q1, Q2 into the staging buffer: tensor m, channel tile t, face f -> slot (m, 2t + f)
    CircularBuffer stg(cb_stage), cw(cb_w), csel(cb_sel), co(cb_onw);
    stg.reserve_back(4);
    cw.reserve_back(4);
    csel.reserve_back(16);
    co.reserve_back(4);
    const uint32_t s0 = stg.get_write_ptr();
    for (uint32_t t = 0; t < 16; t++) {
        const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 8 * hk + (t - 8));
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512, slot = (2 * t + f) * 64;
            noc_async_read(mega.get_noc_addr(c, off), s0 + slot, 64);
            noc_async_read(q0.get_noc_addr(c, off), s0 + tb + slot, 64);
            noc_async_read(q1.get_noc_addr(c, off), s0 + 2 * tb + slot, 64);
            noc_async_read(q2.get_noc_addr(c, off), s0 + 3 * tb + slot, 64);
        }
    }
    for (uint32_t t = 0; t < 4; t++) {
        noc.async_read(wp, cw, tb, {.page_id = 4 * hk + t}, {.offset_bytes = t * tb});
        noc.async_read(onw, co, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    for (uint32_t t = 0; t < 16; t++) {
        noc.async_read(sel, csel, tb, {.page_id = t}, {.offset_bytes = t * tb});
    }
    // packed tiles: x (cb_x, 1 tile) and Q0..Q2 (cb_q, 3 tiles); rows 0..15 (faces 0, 1) are all copied below,
    // rows 16..31 (faces 2, 3: the second half of the tile) are zeroed while the reads fly
    CircularBuffer cx(cb_x), cq(cb_q);
    cx.reserve_back(1);
    cq.reserve_back(3);
    volatile tt_l1_ptr uint32_t* px = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cx.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cq.get_write_ptr());
    for (uint32_t m = 0; m < 4; m++) {
        volatile tt_l1_ptr uint32_t* p = m == 0 ? px : pq + (m - 1) * (tb / 4);
        for (uint32_t k = tb / 8; k < tb / 4; k++) {
            p[k] = 0;
        }
    }
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* sg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s0);
    for (uint32_t m = 0; m < 4; m++) {
        volatile tt_l1_ptr uint32_t* dst = m == 0 ? px : pq + (m - 1) * (tb / 4);
        for (uint32_t t = 0; t < 16; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                // face f row t of the packed tile (8 words) <- row 0 of the chunk (the first 32 bytes of the slot)
                volatile tt_l1_ptr uint32_t* d = dst + (f * 512 + t * 32) / 4;
                volatile tt_l1_ptr uint32_t* s = sg + (m * tb + (2 * t + f) * 64) / 4;
                for (uint32_t w = 0; w < 8; w++) {
                    d[w] = s[w];
                }
            }
        }
    }
    co.push_back(4);
    cx.push_back(1);
    cq.push_back(3);
    cw.push_back(4);
    csel.push_back(16);

    read_n(mega, cb_gate, 258 + 8 * hk, 8);

    // a / b tiles of all 32 heads (mega tiles 256, 257) into the scratch CB (2 tiles, never pushed)
    CircularBuffer scr(cb_scratch);
    scr.reserve_back(2);
    noc.async_read(mega, scr, tb, {.page_id = 256}, {.offset_bytes = 0});
    noc.async_read(mega, scr, tb, {.page_id = 257}, {.offset_bytes = tb});
    noc.async_read_barrier();
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

    for (uint32_t j = 0; j < 2; j++) {
        const uint32_t hv = 2 * hk + j;
        put_scalar(cb_a, ab[row0(hv)]);
        put_scalar(cb_b, ab[1024 + row0(hv)]);
        read_n(dtb, cb_dtb, hv, 1);
        read_n(aneg, cb_aneg, hv, 1);
        read_n(st, cb_s, hv * 16, 16);
    }
}
