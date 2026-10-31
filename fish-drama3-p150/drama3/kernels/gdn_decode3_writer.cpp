// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode v3 (one core per value head), writer. The new partial sums come as 3 packed tiles
// (row t = channel tile t): row t goes back to row 0 of channel tile c in place, as two 64-byte chunks (rows 0-1 of
// faces 0 and 1, row 1 = zero padding) assembled in L1 by NOC copies; the shared q / k rows only from the even core of a key head, once
// the odd core has read them (semaphore). Then the new state and the output o of value head hv.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = 13, cb_out = 16, cb_state = 17, cb_qn = 18;

void kernel_main() {
    constexpr auto q0_a = TensorAccessorArgs<1>();
    constexpr auto q1_a = TensorAccessorArgs<q0_a.next_compile_time_args_offset()>();
    constexpr auto q2_a = TensorAccessorArgs<q1_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<q2_a.next_compile_time_args_offset()>();
    constexpr auto o_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();

    const uint32_t hv = get_arg_val<uint32_t>(0);
    const uint32_t hk = hv / 2;
    constexpr uint32_t SEM = get_compile_time_arg_val(0);
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
    CircularBuffer cqn(cb_qn), cst(cb_state), cout(cb_out), stg(cb_stage);

    cqn.wait_front(3);
    stg.reserve_back(3);
    const uint32_t src = cqn.get_read_ptr(), s0 = stg.get_write_ptr();
    // staging slots (tensor m, tile t, face f -> slot m * 32 + 2t + f, 64 bytes): zero, then row t of the packed
    // tile into the first 32 bytes (row 0); the second half stays 0 (the padding row 1)
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t k = 0; k < 3 * tb; k += 512) {
        noc_async_read(zeros, s0 + k, 512);
    }
    noc_async_read_barrier();
    for (uint32_t m = 0; m < 3; m++) {
        for (uint32_t t = 0; t < 12; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(get_noc_addr(src + m * tb + f * 512 + t * 32), s0 + m * tb + (2 * t + f) * 64, 32);
            }
        }
    }
    noc_async_read_barrier();
    cqn.pop_front(3);
    if (!(hv & 1)) {  // the shared q / k rows: wait until the odd core of this key head has read the old values
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    for (uint32_t t = (hv & 1) ? 8 : 0; t < 12; t++) {
        const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 4 * hv + (t - 8));
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t off = f * 512, slot = (2 * t + f) * 64;
            noc_async_write(s0 + slot, q0.get_noc_addr(c, off), 64);
            noc_async_write(s0 + tb + slot, q1.get_noc_addr(c, off), 64);
            noc_async_write(s0 + 2 * tb + slot, q2.get_noc_addr(c, off), 64);
        }
    }
    noc_async_write_barrier();

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
