// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_FIR_K=1): GDN prefill FIR conv, writer: W output tiles per unit (column tile ct = c * W + j).
// KT = 0: one output [T, D], tile r * NT + ct. KT > 0 (QWEN36_GDN_FUSE2): the conv output split into q / k / v
// (columns [0, KT), [KT, 2 KT), [2 KT, NT) in tiles) as three tensors, as the slices in forward_prefill.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t KT = get_compile_time_arg_val(2);
    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto k_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t u1 = get_arg_val<uint32_t>(2);
    const uint32_t NR = get_arg_val<uint32_t>(3);
    const uint32_t k_addr = get_arg_val<uint32_t>(4);
    const uint32_t v_addr = get_arg_val<uint32_t>(5);
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    const auto kk = TensorAccessor(k_a, k_addr, tb);
    const auto vv = TensorAccessor(v_a, v_addr, tb);
    CircularBuffer co(cb_out);
    for (uint32_t u = u0; u < u1; u++) {
        const uint32_t c = u / NR, r = u % NR;
        co.wait_front(W);
        const uint32_t p = co.get_read_ptr();
        for (uint32_t j = 0; j < W; j++) {
            const uint32_t ct = c * W + j;
            uint64_t a;
            if constexpr (KT == 0) {
                a = o.get_noc_addr(r * NT + ct, 0);
            } else {
                if (ct < KT) {
                    a = o.get_noc_addr(r * KT + ct, 0);
                } else if (ct < 2 * KT) {
                    a = kk.get_noc_addr(r * KT + ct - KT, 0);
                } else {
                    a = vv.get_noc_addr(r * (NT - 2 * KT) + ct - 2 * KT, 0);
                }
            }
            noc_async_write(p + j * tb, a, tb);
        }
        noc_async_write_barrier();
        co.pop_front(W);
    }
}
