// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: residual add + Snake1d, writer: per item the fp32 sum (cb 17) and the bf16 snake (cb 16), both
// to page r * CT + c of their outputs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_sum = 17;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t tbo = get_tile_size(cb_out), tbs = get_tile_size(cb_sum);
    const auto O = TensorAccessor(o_a, get_arg_val<uint32_t>(2), tbo);
    const auto S = TensorAccessor(s_a, get_arg_val<uint32_t>(3), tbs);
    Noc noc;
    CircularBuffer co(cb_out), cs(cb_sum);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = (i0 + i) / R, r = (i0 + i) % R;
        cs.wait_front(1);
        co.wait_front(1);
        auto ss = use<CircularBuffer::AddrSelector::READ_PTR>(cs);
        auto so = use<CircularBuffer::AddrSelector::READ_PTR>(co);
        noc.async_write(ss, S, tbs, {.offset_bytes = 0}, {.page_id = r * CT + c});
        noc.async_write(so, O, tbo, {.offset_bytes = 0}, {.page_id = r * CT + c});
        noc.async_write_barrier();
        cs.pop_front(1);
        co.pop_front(1);
    }
}
