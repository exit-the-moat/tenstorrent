// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_GDN_FUSE2=1): GDN prefill output gate, compute: out = n * silu(z) per tile, with the stock ops' math:
// n is fp32 (the rms-normed chunk output), z bf16. ttnn.silu(z) runs in a 16-bit dest: calculate_silu<false> (its
// sigmoid variant and the round-to-nearest bf16 result) is called explicitly here in the fp32 dest; ttnn.multiply(n,
// silu_z) runs as an SFPU binary mul with an fp32 dest and both inputs unpacked to dest in fp32 (as binary_ng). So the
// output (fp32) is bit-identical to multiply(concat_heads(n), silu(z)).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_n = 0, cb_z = 1, cb_out = 16;

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_z, cb_n, cb_out);
    CircularBuffer cn(cb_n), cz(cb_z), co(cb_out);
    constexpr uint32_t G = 2;  // tiles per dest pass (fp32 dest: 4 tiles = 2 z + 2 n), one format switch per pass
    for (uint32_t t = 0; t < nt; t += G) {
        const uint32_t g = (nt - t) < G ? (nt - t) : G;
        cn.wait_front(g);
        cz.wait_front(g);
        co.reserve_back(g);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_z);
        copy_tile_to_dst_init_short(cb_z);
        for (uint32_t i = 0; i < g; i++) {
            copy_tile(cb_z, i, G + i);
        }
        silu_tile_init();
        for (uint32_t i = 0; i < g; i++) {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (false /* bf16 op: 16-bit dest math */, 8), G + i,
                VectorMode::RC));
        }
        reconfig_data_format_srca(cb_n);
        copy_tile_to_dst_init_short(cb_n);
        for (uint32_t i = 0; i < g; i++) {
            copy_tile(cb_n, i, i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < g; i++) {
            mul_binary_tile(i, G + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < g; i++) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        co.push_back(g);
        cn.pop_front(g);
        cz.pop_front(g);
    }
}
