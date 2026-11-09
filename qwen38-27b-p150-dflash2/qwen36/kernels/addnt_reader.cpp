// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 (Exit the Moat, Episode 2): TP decode residual add + RMSNorm in one program (QWEN36_ADDNORM_TP=1, on the
// QWEN36_CCL_AR replicated residual stream), reader of core k. x (the residual), d (the all-reduced mixer / MLP output)
// are width-sharded on the same NC cores (PER tiles each, one tile row = up to 32 users): local L1 copies into the
// CBs; plus this core's PER tiles of the row-replicated norm weight (interleaved DRAM) and one all-ones tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_d = 1, cb_w = 2, cb_ones = 3;

void kernel_main() {
    constexpr uint32_t PER = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_D = get_compile_time_arg_val(1);
    constexpr auto one_a = TensorAccessorArgs<2>();
    constexpr auto w_a = TensorAccessorArgs<one_a.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t one_addr = get_arg_val<uint32_t>(3);
    const uint32_t k = get_arg_val<uint32_t>(4);
    constexpr uint32_t tb = 2048;
    const auto one = TensorAccessor(one_a, one_addr, tb);
    const auto wt = TensorAccessor(w_a, w_addr, tb);
    Noc noc;
    CircularBuffer cx(cb_x), cd(cb_d), cw(cb_w), co(cb_ones);
    cx.reserve_back(PER);
    noc_async_read(get_noc_addr(x_addr), cx.get_write_ptr(), PER * tb);
    if constexpr (HAS_D) {
        cd.reserve_back(PER);
        noc_async_read(get_noc_addr(d_addr), cd.get_write_ptr(), PER * tb);
    }
    cw.reserve_back(PER);
    for (uint32_t i = 0; i < PER; i++) {  // the row-replicated weight tiles of this core (interleaved DRAM)
        noc_async_read(wt.get_noc_addr(k * PER + i, 0), cw.get_write_ptr() + i * tb, tb);
    }
    co.reserve_back(1);
    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cx.push_back(PER);
    if constexpr (HAS_D) {
        cd.push_back(PER);
    }
    co.push_back(1);
    cw.push_back(PER);
}
