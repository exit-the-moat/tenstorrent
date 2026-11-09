// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused gated-attention post-SDPA step for B sequences (TP path), reader. Core j (of 8) reads column tile j of
// the SDPA output [1, B, NH, 256] and of the packed gate (same layout) for every sequence b (pages 8 b + j).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_a = 0, cb_g = 1;

void kernel_main() {
    constexpr auto a_a = TensorAccessorArgs<0>();
    constexpr auto g_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t B = get_arg_val<uint32_t>(3);
    constexpr uint32_t tb = 2048;
    const auto a = TensorAccessor(a_a, a_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);
    Noc noc;
    CircularBuffer ca(cb_a), cg(cb_g);
    cg.reserve_back(B);
    ca.reserve_back(B);
    for (uint32_t b = 0; b < B; b++) {
        noc.async_read(g, cg, tb, {.page_id = 8 * b + j}, {.offset_bytes = b * tb});
        noc.async_read(a, ca, tb, {.page_id = 8 * b + j}, {.offset_bytes = b * tb});
    }
    noc.async_read_barrier();
    cg.push_back(B);
    ca.push_back(B);
}
