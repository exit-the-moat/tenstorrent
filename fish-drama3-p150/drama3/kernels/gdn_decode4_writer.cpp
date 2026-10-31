// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: fused GDN decode v4 (B sequences, kernels/gdn_decode4_reader.cpp for the layouts), writer. Per
// sequence b of this core: the new partial sums to row 2b of the [1, 2B, 8192] Q tensors (64-byte chunks: row 2b
// and the zero row 2b + 1), the shared q / k rows only from the even core once the odd one has read sequence b
// (semaphore count), the new state to sequence b's pages, and o's row 0 of each output tile to row b of the
// [1, B, 4096] output (32-byte writes; a fresh L1 tensor per call, the g = 0 cores zero its padding rows).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = 13, cb_out = 16, cb_state = 17, cb_qn = 18;

inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr auto q0_a = TensorAccessorArgs<1>();
    constexpr auto q1_a = TensorAccessorArgs<q0_a.next_compile_time_args_offset()>();
    constexpr auto q2_a = TensorAccessorArgs<q1_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<q2_a.next_compile_time_args_offset()>();
    constexpr auto o_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();

    const uint32_t hv = get_arg_val<uint32_t>(0);
    const uint32_t hk = hv / 2;
    constexpr uint32_t SEM = get_compile_time_arg_val(0);
    const uint32_t g0 = get_arg_val<uint32_t>(1);
    const uint32_t G = get_arg_val<uint32_t>(2);
    const uint32_t B = get_arg_val<uint32_t>(3);
    const uint32_t q0_addr = get_arg_val<uint32_t>(4);
    const uint32_t q1_addr = get_arg_val<uint32_t>(5);
    const uint32_t q2_addr = get_arg_val<uint32_t>(6);
    const uint32_t s_addr = get_arg_val<uint32_t>(7);
    const uint32_t o_addr = get_arg_val<uint32_t>(8);

    const uint32_t tb = get_tile_size(cb_out);
    const auto q0 = TensorAccessor(q0_a, q0_addr, tb);
    const auto q1 = TensorAccessor(q1_a, q1_addr, tb);
    const auto q2 = TensorAccessor(q2_a, q2_addr, tb);
    const auto st = TensorAccessor(s_a, s_addr, tb);
    const auto o = TensorAccessor(o_a, o_addr, tb);

    Noc noc;
    CircularBuffer cqn(cb_qn), cst(cb_state), cout(cb_out), stg(cb_stage);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    uint32_t done = 0;
    if (g0 == 0) {  // the output is a fresh L1 tensor: zero its padding rows B..31 of this head (4 tiles)
        for (uint32_t t = 0; t < 4; t++) {
            for (uint32_t r = B; r < 16; r++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(hv * 4 + t, rowoff(r, f)), 32);
                }
            }
            noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(hv * 4 + t, 1024), 512);
            noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(hv * 4 + t, 1536), 512);
        }
        noc_async_write_barrier();
    }

    for (uint32_t b = g0; b < B; b += G) {
        const uint32_t qb = rowoff(2 * b, 0);
        cqn.wait_front(3);
        stg.reserve_back(3);
        const uint32_t src = cqn.get_read_ptr(), s0 = stg.get_write_ptr();
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
        done++;
        if (!(hv & 1)) {  // the shared q / k rows: wait until the odd core has read this sequence's old values
            noc_semaphore_wait_min(sem, done);
        }
        for (uint32_t t = (hv & 1) ? 8 : 0; t < 12; t++) {
            const uint32_t c = t < 4 ? 4 * hk + t : (t < 8 ? 64 + 4 * hk + (t - 4) : 128 + 4 * hv + (t - 8));
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t slot = (2 * t + f) * 64;
                noc_async_write(s0 + slot, q0.get_noc_addr(c, qb + f * 512), 64);
                noc_async_write(s0 + tb + slot, q1.get_noc_addr(c, qb + f * 512), 64);
                noc_async_write(s0 + 2 * tb + slot, q2.get_noc_addr(c, qb + f * 512), 64);
            }
        }
        noc_async_write_barrier();

        cst.wait_front(16);
        auto ss = use<CircularBuffer::AddrSelector::READ_PTR>(cst);
        for (uint32_t t = 0; t < 16; t++) {
            noc.async_write(ss, st, tb, {.offset_bytes = t * tb}, {.page_id = (b * 32 + hv) * 16 + t});
        }
        noc.async_write_barrier();
        cst.pop_front(16);
        cout.wait_front(4);
        const uint32_t so = cout.get_read_ptr();
        for (uint32_t t = 0; t < 4; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(so + t * tb + f * 512, o.get_noc_addr(hv * 4 + t, rowoff(b, f)), 32);
            }
        }
        noc.async_write_barrier();
        cout.pop_front(4);
    }
    if (!(hv & 1)) {
        noc_semaphore_set(sem, 0);
    }
}
