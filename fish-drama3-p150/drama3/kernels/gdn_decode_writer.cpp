// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode, writer. Per core (key head hk): the updated conv partial sums Q0..Q2
// of its 16 channel tiles (in place), then per value head j the new state (16 tiles, in place) and the gated,
// normed output o_j (4 tiles of the [1, 1, 32 * 128] o row).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_state = 17, cb_qn = 18;

void kernel_main() {
    constexpr auto q0_a = TensorAccessorArgs<0>();
    constexpr auto q1_a = TensorAccessorArgs<q0_a.next_compile_time_args_offset()>();
    constexpr auto q2_a = TensorAccessorArgs<q1_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<q2_a.next_compile_time_args_offset()>();
    constexpr auto o_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();

    const uint32_t hk = get_arg_val<uint32_t>(0);
    const uint32_t q0_addr = get_arg_val<uint32_t>(1);
    const uint32_t q1_addr = get_arg_val<uint32_t>(2);
    const uint32_t q2_addr = get_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_arg_val<uint32_t>(4);
    const uint32_t o_addr = get_arg_val<uint32_t>(5);

    const uint32_t tb = get_tile_size(cb_out);
    const auto q0 = TensorAccessor(q0_a, q0_addr, tb);
    const auto q1 = TensorAccessor(q1_a, q1_addr, tb);
    const auto q2 = TensorAccessor(q2_a, q2_addr, tb);
    const auto st = TensorAccessor(s_a, s_addr, tb);
    const auto o = TensorAccessor(o_a, o_addr, tb);

    Noc noc;
    CircularBuffer cqn(cb_qn), cst(cb_state), cout(cb_out);

    for (uint32_t t = 0; t < 16; t++) {
        const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 8 * hk + (t - 8));
        cqn.wait_front(3);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cqn);
        noc.async_write(src, q0, tb, {.offset_bytes = 0}, {.page_id = c});
        noc.async_write(src, q1, tb, {.offset_bytes = tb}, {.page_id = c});
        noc.async_write(src, q2, tb, {.offset_bytes = 2 * tb}, {.page_id = c});
        noc.async_write_barrier();
        cqn.pop_front(3);
    }
    for (uint32_t j = 0; j < 2; j++) {
        const uint32_t hv = 2 * hk + j;
        cst.wait_front(16);
        auto ss = use<CircularBuffer::AddrSelector::READ_PTR>(cst);
        for (uint32_t t = 0; t < 16; t++) {
            noc.async_write(ss, st, tb, {.offset_bytes = t * tb}, {.page_id = hv * 16 + t});
        }
        noc.async_write_barrier();
        cst.pop_front(16);
        cout.wait_front(4);
        auto so = use<CircularBuffer::AddrSelector::READ_PTR>(cout);
        for (uint32_t t = 0; t < 4; t++) {
            noc.async_write(so, o, tb, {.offset_bytes = t * tb}, {.page_id = hv * 4 + t});
        }
        noc.async_write_barrier();
        cout.pop_front(4);
    }
}
