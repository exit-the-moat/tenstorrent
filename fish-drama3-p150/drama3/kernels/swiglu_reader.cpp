// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token SwiGLU over [up | gate] slots, reader. Input [1, S, 1, 2N] (each slot one tile row),
// work items i0 .. i0 + n of S * N/32: per item the up tile (page s * 2NT + c) and the gate tile (page s * 2NT + NT + c).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_up = 0, cb_gate = 1;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);  // output tiles per slot
    constexpr auto in_a = TensorAccessorArgs<1>();
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t in_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(cb_up);
    const auto in = TensorAccessor(in_a, in_addr, tb);
    Noc noc;
    CircularBuffer cu(cb_up), cg(cb_gate);
    cu.reserve_back(n);
    cg.reserve_back(n);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t s = (i0 + i) / NT, c = (i0 + i) % NT;
        noc.async_read(in, cu, tb, {.page_id = s * 2 * NT + c}, {.offset_bytes = i * tb});
        noc.async_read(in, cg, tb, {.page_id = s * 2 * NT + NT + c}, {.offset_bytes = i * tb});
    }
    noc.async_read_barrier();
    cu.push_back(n);
    cg.push_back(n);
}
