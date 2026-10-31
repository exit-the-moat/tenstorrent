// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token MoE combine, writer: this core's n output tiles from page t0 on.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto o_a = TensorAccessorArgs<0>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    Noc noc;
    CircularBuffer co(cb_out);
    for (uint32_t i = 0; i < n; i++) {
        co.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(co), o, tb, {.offset_bytes = 0}, {.page_id = t0 + i});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}
