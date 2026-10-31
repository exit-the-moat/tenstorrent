// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: copy one sequence's decode state into a slot of the batched state (continuous batching), data
// movement only. Mode 0: whole bf16 tiles, src pages p0 + i -> dst pages d0 + i. Mode 1: the 64-byte row-pair chunks
// (rows 0-1 of faces 0 and 1) of each src tile -> rows (dr, dr + 1) of the same dst tile (the [1, 2B, D] GDN conv
// partial sums: dr = 2b). This core copies items [i0, i0 + n).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr auto s_a = TensorAccessorArgs<1>();
    constexpr auto d_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t i0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t p0 = get_arg_val<uint32_t>(4);
    const uint32_t d0 = get_arg_val<uint32_t>(5);
    const uint32_t dr = get_arg_val<uint32_t>(6);
    const uint32_t tb = get_tile_size(0);
    const auto src = TensorAccessor(s_a, s_addr, tb);
    const auto dst = TensorAccessor(d_a, d_addr, tb);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t l1 = cb.get_write_ptr();
    for (uint32_t i = i0; i < i0 + n; i++) {
        if (MODE == 0) {
            noc_async_read(src.get_noc_addr(p0 + i, 0), l1, tb);
            noc_async_read_barrier();
            noc_async_write(l1, dst.get_noc_addr(d0 + i, 0), tb);
        } else {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(src.get_noc_addr(p0 + i, rowoff(0, f)), l1 + f * 64, 64);
            }
            noc_async_read_barrier();
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(l1 + f * 64, dst.get_noc_addr(d0 + i, rowoff(dr, f)), 64);
            }
        }
        noc_async_write_barrier();
    }
}
