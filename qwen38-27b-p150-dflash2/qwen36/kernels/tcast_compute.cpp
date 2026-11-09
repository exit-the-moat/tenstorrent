// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_PF_TCAST=1): bf16 -> fp32 typecast, compute: unpack to an fp32 dest (exact widening), pack fp32.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0);
    constexpr uint32_t G = 4;
    compute_kernel_hw_startup(0, 16);
    copy_tile_to_dst_init_short(0);
    CircularBuffer ci(0), co(16);
    for (uint32_t t = 0; t < nt; t += G) {
        const uint32_t g = (nt - t) < G ? (nt - t) : G;
        ci.wait_front(g);
        co.reserve_back(g);
        tile_regs_acquire();
        for (uint32_t i = 0; i < g; i++) {
            copy_tile(0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < g; i++) {
            pack_tile(i, 16);
        }
        tile_regs_release();
        co.push_back(g);
        ci.pop_front(g);
    }
}
