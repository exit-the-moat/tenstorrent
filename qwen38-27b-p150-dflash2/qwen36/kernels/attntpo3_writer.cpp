// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 gdn: fused gated-attention post-SDPA step for B sequences (TP path), writer for QWEN36_ATTN_POST_ROWS: core
// (j, group) handles the sequences [B0, B0 + NB) of column tile j. Output tile 8h + j of the [1, B, NH * 256] wo input
// holds, in row b, row h of sequence b's product tile: written straight from the product tiles as 32-byte row chunks
// (no build buffer, no local copies; the cores of one tile write disjoint rows of the L1 output). The last group also
// writes the rows B..31 of its tiles as zeros. Same output as attntpo_writer.cpp.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

constexpr uint32_t rowoff(uint32_t r, uint32_t c) { return (r >> 4) * 1024 + c * 512 + (r & 15) * 32; }

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t B = get_arg_val<uint32_t>(2);
    const uint32_t b0 = get_arg_val<uint32_t>(3), nb = get_arg_val<uint32_t>(4), last = get_arg_val<uint32_t>(5);
    constexpr uint32_t tb = 2048;
    const auto o = TensorAccessor(o_a, o_addr, tb);
    if (last) {  // rows B..31 of the output tiles of column j
        for (uint32_t h = 0; h < NH; h++) {
            if (B < 16) {
                for (uint32_t c = 0; c < 2; c++) {
                    noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(B, c)), (16 - B) * 32);
                }
            }
            const uint32_t r2 = B > 16 ? B : 16;
            for (uint32_t c = 0; c < 2; c++) {
                if (r2 < 32) {
                    noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r2, c)), (32 - r2) * 32);
                }
            }
        }
    }
    CircularBuffer co(cb_out);
    co.wait_front(nb);
    const uint32_t src = co.get_read_ptr();
    for (uint32_t k = 0; k < nb; k++) {
        const uint32_t b = b0 + k;
        for (uint32_t h = 0; h < NH; h++) {
            for (uint32_t c = 0; c < 2; c++) {
                // r64f: a sequence >= 32 writes its tile row one page-row (NH * 8 column tiles of [1, B, NH * 256])
                // further, and its row offset is taken modulo 32. Exactly 0 / no change for rows < 32.
                noc_async_write(src + k * tb + rowoff(h, c),
                                o.get_noc_addr(8 * h + j + (b >> 5) * (NH * 8), rowoff(b & 31u, c)), 32);
            }
        }
    }
    noc_async_write_barrier();
    co.pop_front(nb);
}
