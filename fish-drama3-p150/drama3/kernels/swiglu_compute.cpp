// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: one-token SwiGLU, compute: out = up * silu(gate) per tile (SFPU silu and an fp32 SFPU multiply in DST;
// an FPU dest-reuse multiply would truncate silu(gate) on its way into a source register).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_up = 0, cb_gate = 1, cb_out = 16;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);  // work items of this core
    compute_kernel_hw_startup(cb_gate, cb_up, cb_out);
    CircularBuffer cu(cb_up), cg(cb_gate), co(cb_out);
    cu.wait_front(N);
    cg.wait_front(N);
    co.reserve_back(N);
    for (uint32_t i = 0; i < N; i++) {
        tile_regs_acquire();
        copy_init(cb_gate);
        copy_tile(cb_gate, i, 0);
        silu_tile_init();
        silu_tile(0);
        copy_init(cb_up);
        copy_tile(cb_up, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    co.push_back(N);
    cu.pop_front(N);
    cg.pop_front(N);
}
