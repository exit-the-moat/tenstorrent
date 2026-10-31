// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode (between the mega projection and o_proj), reader.
// One core per key head hk = 2 value heads (2hk, 2hk+1), so no two cores share a conv channel.
// Pushes, in the order the compute kernel consumes them:
//   16 conv groups (channel tiles q 4hk.., k 64+4hk.., v 128+8hk..): x (mega tile), Q0..Q2 (partial sums), w0..w3
//   8 gate tiles (mega 258+8hk..), then per value head j: a_j, b_j (scalars at [0,0], cut out of mega tiles 256 / 257),
//   dt_bias_j, A_neg_j (host-built per-head tiles), the state S_j (16 tiles); the o-norm weight (4 tiles) first.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_gate = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_s = 8,
                   cb_onw = 9, cb_scratch = 10;

void kernel_main() {
    constexpr auto mega_a = TensorAccessorArgs<0>();
    constexpr auto q0_a = TensorAccessorArgs<mega_a.next_compile_time_args_offset()>();
    constexpr auto q1_a = TensorAccessorArgs<q0_a.next_compile_time_args_offset()>();
    constexpr auto q2_a = TensorAccessorArgs<q1_a.next_compile_time_args_offset()>();
    constexpr auto w0_a = TensorAccessorArgs<q2_a.next_compile_time_args_offset()>();
    constexpr auto w1_a = TensorAccessorArgs<w0_a.next_compile_time_args_offset()>();
    constexpr auto w2_a = TensorAccessorArgs<w1_a.next_compile_time_args_offset()>();
    constexpr auto w3_a = TensorAccessorArgs<w2_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<w3_a.next_compile_time_args_offset()>();
    constexpr auto dtb_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto aneg_a = TensorAccessorArgs<dtb_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<aneg_a.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hk = get_arg_val<uint32_t>(i++);
    const uint32_t mega_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);

    const uint32_t tb = get_tile_size(cb_x);  // bf16; every tensor here is bf16 TILE
    const auto mega = TensorAccessor(mega_a, mega_addr, tb);
    const auto q0 = TensorAccessor(q0_a, q0_addr, tb);
    const auto q1 = TensorAccessor(q1_a, q1_addr, tb);
    const auto q2 = TensorAccessor(q2_a, q2_addr, tb);
    const auto w0 = TensorAccessor(w0_a, w0_addr, tb);
    const auto w1 = TensorAccessor(w1_a, w1_addr, tb);
    const auto w2 = TensorAccessor(w2_a, w2_addr, tb);
    const auto w3 = TensorAccessor(w3_a, w3_addr, tb);
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

    read_n(onw, cb_onw, 0, 4);

    // conv groups: x, Q0, Q1, Q2 (into cb_q, 3 tiles), w0..w3 (into cb_w, 4 tiles)
    for (uint32_t t = 0; t < 16; t++) {
        const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 8 * hk + (t - 8));
        read_n(mega, cb_x, c, 1);
        CircularBuffer cq(cb_q);
        cq.reserve_back(3);
        noc.async_read(q0, cq, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(q1, cq, tb, {.page_id = c}, {.offset_bytes = tb});
        noc.async_read(q2, cq, tb, {.page_id = c}, {.offset_bytes = 2 * tb});
        CircularBuffer cw(cb_w);
        cw.reserve_back(4);
        noc.async_read(w0, cw, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(w1, cw, tb, {.page_id = c}, {.offset_bytes = tb});
        noc.async_read(w2, cw, tb, {.page_id = c}, {.offset_bytes = 2 * tb});
        noc.async_read(w3, cw, tb, {.page_id = c}, {.offset_bytes = 3 * tb});
        noc.async_read_barrier();
        cq.push_back(3);
        cw.push_back(4);
    }

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
