// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: residual add + Snake1d, reader: as snake_reader.cpp plus the y tile (same page as x) per item.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_a = 1, cb_i = 2, cb_y = 3;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr auto x_a = TensorAccessorArgs<2>();
    constexpr auto y_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto a_a = TensorAccessorArgs<y_a.next_compile_time_args_offset()>();
    constexpr auto i_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_x);
    const auto X = TensorAccessor(x_a, get_arg_val<uint32_t>(2), tb);
    const auto Y = TensorAccessor(y_a, get_arg_val<uint32_t>(3), tb);
    const auto A = TensorAccessor(a_a, get_arg_val<uint32_t>(4), tb);
    const auto I = TensorAccessor(i_a, get_arg_val<uint32_t>(5), tb);
    Noc noc;
    CircularBuffer cx(cb_x), cy(cb_y), ca(cb_a), ci(cb_i);
    uint32_t cprev = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = (i0 + i) / R, r = (i0 + i) % R;
        if (c != cprev) {
            ca.reserve_back(1);
            ci.reserve_back(1);
            noc.async_read(A, ca, tb, {.page_id = c}, {.offset_bytes = 0});
            noc.async_read(I, ci, tb, {.page_id = c}, {.offset_bytes = 0});
            noc.async_read_barrier();
            ca.push_back(1);
            ci.push_back(1);
            cprev = c;
        }
        cx.reserve_back(1);
        cy.reserve_back(1);
        noc.async_read(X, cx, tb, {.page_id = r * CT + c}, {.offset_bytes = 0});
        noc.async_read(Y, cy, tb, {.page_id = r * CT + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cx.push_back(1);
        cy.push_back(1);
    }
}
