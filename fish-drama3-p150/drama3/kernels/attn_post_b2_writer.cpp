// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: attn_post_b_writer.cpp with several rows per core (B > cores / 8; B <= 32): rows b = g0, g0 + G, ..
// < B. The g0 = 0 cores zero the padding rows B..31 row by row: rows >= 16 may be other cores' sequences, so faces
// 2 / 3 are not zeroed as a whole.

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
    const uint32_t g0 = get_arg_val<uint32_t>(2);
    const uint32_t G = get_arg_val<uint32_t>(3);
    const uint32_t B = get_arg_val<uint32_t>(4);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out);
    if (g0 == 0) {
        for (uint32_t h = 0; h < NH; h++) {
            for (uint32_t r = B; r < 32; r++) {
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 0)), 32);
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 1)), 32);
            }
        }
    }
    for (uint32_t b = g0; b < B; b += G) {
        co.wait_front(1);
        const uint32_t src = co.get_read_ptr();
        for (uint32_t h = 0; h < NH; h++) {
            noc_async_write(src + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 0)), 32);
            noc_async_write(src + 512 + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 1)), 32);
        }
        noc_async_write_barrier();
        co.pop_front(1);
    }
    noc_async_write_barrier();
}
