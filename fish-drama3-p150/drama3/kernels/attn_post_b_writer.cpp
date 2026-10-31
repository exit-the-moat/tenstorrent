// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: fused gated-attention post-SDPA step for B decode rows, writer. Row h of the packed product tile
// (column tile j of sequence b) goes to row b of output tile 8 h + j of the [1, B, 16 * 256] o_proj input, as the two
// 32-byte row chunks of faces 0 / 1 (the output is a fresh L1 tensor: L1 takes 16-byte aligned writes); the b = 0
// cores also zero the padding rows B..31 of their output tiles.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t b = get_arg_val<uint32_t>(2);
    const uint32_t B = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out);
    if (b == 0) {
        for (uint32_t h = 0; h < NH; h++) {
            for (uint32_t r = B; r < 16; r++) {
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 0)), 32);
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 1)), 32);
            }
            noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, 1024), 512);
            noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, 1536), 512);
        }
    }
    co.wait_front(1);
    const uint32_t src = co.get_read_ptr();
    for (uint32_t h = 0; h < NH; h++) {
        noc_async_write(src + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 0)), 32);
        noc_async_write(src + 512 + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 1)), 32);
    }
    noc_async_write_barrier();
    co.pop_front(1);
}
