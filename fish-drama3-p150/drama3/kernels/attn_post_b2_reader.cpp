// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: attn_post_b_reader.cpp with several rows per core (B > cores / 8): core (j, g0) reads tiles 8 b + j
// of the SDPA output and the packed gate for b = g0, g0 + G, .. < B.

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
    const uint32_t g0 = get_arg_val<uint32_t>(3);
    const uint32_t G = get_arg_val<uint32_t>(4);
    const uint32_t B = get_arg_val<uint32_t>(5);
    const uint32_t tb = get_tile_size(cb_a);
    const auto a = TensorAccessor(a_a, a_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);
    Noc noc;
    CircularBuffer ca(cb_a), cg(cb_g);
    for (uint32_t b = g0; b < B; b += G) {
        ca.reserve_back(1);
        cg.reserve_back(1);
        noc.async_read(a, ca, tb, {.page_id = 8 * b + j}, {.offset_bytes = 0});
        noc.async_read(g, cg, tb, {.page_id = 8 * b + j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        ca.push_back(1);
        cg.push_back(1);
    }
}
