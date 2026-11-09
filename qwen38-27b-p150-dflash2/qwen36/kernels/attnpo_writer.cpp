// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused one-token gated-attention post-SDPA step (port of drama3 attn_post_writer.cpp to up to 32 heads),
// writer: unpacks row h of the packed product tile (column tile j) into row 0 of output tile 8h + j of the
// [1, 1, NH * 256] o_proj input: each output tile is built in L1 (MEM_ZEROS fill + the two 32-byte row chunks, faces
// 0 / 1), then written whole.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_build = 2;

constexpr uint32_t rowoff(uint32_t r, uint32_t c) { return (r >> 4) * 1024 + c * 512 + (r & 15) * 32; }

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out), cb(cb_build);
    cb.reserve_back(NH);
    const uint32_t b0 = cb.get_write_ptr();
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t h = 0; h < NH; h++) {  // everything but row 0 of faces 0 / 1
        const uint32_t p = b0 + h * tb;
        noc_async_read(zeros, p + 32, 480);
        noc_async_read(zeros, p + 512 + 32, 480);
        noc_async_read(zeros, p + 1024, 512);
        noc_async_read(zeros, p + 1536, 512);
    }
    co.wait_front(1);
    const uint32_t src = co.get_read_ptr();
    for (uint32_t h = 0; h < NH; h++) {
        noc_async_read(get_noc_addr(src + rowoff(h, 0)), b0 + h * tb, 32);
        noc_async_read(get_noc_addr(src + rowoff(h, 1)), b0 + h * tb + 512, 32);
    }
    noc_async_read_barrier();
    co.pop_front(1);
    for (uint32_t h = 0; h < NH; h++) {
        noc_async_write(b0 + h * tb, o.get_noc_addr(8 * h + j, 0), tb);
    }
    noc_async_write_barrier();
}
