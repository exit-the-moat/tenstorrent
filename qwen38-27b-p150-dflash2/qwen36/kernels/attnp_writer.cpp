// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused one-token gated-attention pre-SDPA step, writer (as drama3 attn_pre_writer.cpp): the 8 packed [heads,
// 256] tiles to the destination (q: the DRAM SDPA input; k / v: the height-sharded L1 input of paged_update_cache;
// the packed gate: L1), pages 0..7.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto d_a = TensorAccessorArgs<0>();
    const uint32_t d_addr = get_arg_val<uint32_t>(0);
    const uint32_t pg0 = get_arg_val<uint32_t>(1);  // first destination page (sequence b: 8 b)
    constexpr uint32_t tb = 2048;
    const auto d = TensorAccessor(d_a, d_addr, tb);
    Noc noc;
    CircularBuffer co(cb_out);
    co.wait_front(8);
    auto s = use<CircularBuffer::AddrSelector::READ_PTR>(co);
    for (uint32_t t = 0; t < 8; t++) {
        noc.async_write(s, d, tb, {.offset_bytes = t * tb}, {.page_id = pg0 + t});
    }
    noc.async_write_barrier();
    co.pop_front(8);
}
