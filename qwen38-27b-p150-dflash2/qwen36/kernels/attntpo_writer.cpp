// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused gated-attention post-SDPA step for B sequences (TP path), writer of core j: output tile 8h + j of the
// [1, B, NH * 256] wo input holds, in row b, row h of sequence b's product tile (column tile j). Each output tile is
// built whole in L1 (MEM_ZEROS fill + 2 x 32-byte chunks per sequence) and written once: no two cores share a tile.

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
    const uint32_t B = get_arg_val<uint32_t>(2);
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out), cb(cb_build);
    cb.reserve_back(NH);
    const uint32_t b0 = cb.get_write_ptr();
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t h = 0; h < NH; h++) {
        for (uint32_t q = 0; q < tb; q += 512) {
            noc_async_read(zeros, b0 + h * tb + q, 512);
        }
    }
    noc_async_read_barrier();
    co.wait_front(B);
    const uint32_t src = co.get_read_ptr();
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < NH; h++) {
            noc_async_read(get_noc_addr(src + b * tb + rowoff(h, 0)), b0 + h * tb + rowoff(b, 0), 32);
            noc_async_read(get_noc_addr(src + b * tb + rowoff(h, 1)), b0 + h * tb + rowoff(b, 1), 32);
        }
    }
    noc_async_read_barrier();
    co.pop_front(B);
    for (uint32_t h = 0; h < NH; h++) {
        noc_async_write(b0 + h * tb, o.get_noc_addr(8 * h + j, 0), tb);
    }
    noc_async_write_barrier();
}
