// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn: attnp_writer.cpp for QWEN36_ATTN_SPLIT: runtime args 2 / 3 = H0 / HS > 0 -> only the rows [H0, H0 + HS)
// of every tile (H0 + HS <= 16, H0 even: 64-byte aligned row blocks of faces 0 / 1), arg 4 = 1: also faces 2 / 3;
// HS 0 = whole tiles.
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
    const uint32_t h0 = get_arg_val<uint32_t>(2), hs = get_arg_val<uint32_t>(3);
    if (hs == 0) {
        for (uint32_t t = 0; t < 8; t++) {
            noc.async_write(s, d, tb, {.offset_bytes = t * tb}, {.page_id = pg0 + t});
        }
    } else {
        const uint32_t rp = co.get_read_ptr();
        for (uint32_t t = 0; t < 8; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(rp + t * tb + f * 512 + h0 * 32, d.get_noc_addr(pg0 + t, f * 512 + h0 * 32), hs * 32);
            }
            if (get_arg_val<uint32_t>(4)) {  // the last core of the role: also the (zero) rows 16..31
                noc_async_write(rp + t * tb + 1024, d.get_noc_addr(pg0 + t, 1024), 1024);
            }
        }
    }
    noc.async_write_barrier();
    co.pop_front(8);
}
