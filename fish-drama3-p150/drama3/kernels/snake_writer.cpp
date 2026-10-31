// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: fused Snake1d, writer: item i (c = item / R, r = item % R) to bf16 output page r * CT + c.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto O = TensorAccessor(o_a, get_arg_val<uint32_t>(2), tb);
    Noc noc;
    CircularBuffer co(cb_out);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = (i0 + i) / R, r = (i0 + i) % R;
        co.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(co);
        noc.async_write(src, O, tb, {.offset_bytes = 0}, {.page_id = r * CT + c});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}
