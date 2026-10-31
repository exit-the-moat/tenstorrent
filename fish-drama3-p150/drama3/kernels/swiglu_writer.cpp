// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token SwiGLU, writer: items i0 .. i0 + n to output pages s * NT + c (= i0 + i).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto o_a = TensorAccessorArgs<0>();
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    Noc noc;
    CircularBuffer co(cb_out);
    co.wait_front(n);
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(co);
    for (uint32_t i = 0; i < n; i++) {
        noc.async_write(src, o, tb, {.offset_bytes = i * tb}, {.page_id = i0 + i});
    }
    noc.async_write_barrier();
    co.pop_front(n);
}
