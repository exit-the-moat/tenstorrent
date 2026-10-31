// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: fused Snake1d for the codec decoder (DRAMA3_CODEC_SNAKE_KERNEL=1), compute. Per x tile (fp32), the
// default path's three ops in DST without the round trips through DRAM: multiply(x, alpha) with the post-activations
// sin and square (binary_ng's SFPU mul_binary_tile, sin_tile, square_tile), addcmul(x, t, inv, value=1)
// (addcmul_tile<Float32>: x + (1 * t) * inv) and the typecast to bf16 (typecast_tile<Float32, Float16_b>) for the
// conv that reads it. Items in column-major order (c = item / R): the alpha / inv tiles of column c stay in their CBs.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_a = 1, cb_i = 2, cb_out = 16;
constexpr uint32_t F32 = static_cast<uint32_t>(DataFormat::Float32), BF16 = static_cast<uint32_t>(DataFormat::Float16_b);

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);  // tile rows of x
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_x, cb_a, cb_out);
    CircularBuffer cx(cb_x), ca(cb_a), ci(cb_i), co(cb_out);
    uint32_t cprev = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = (i0 + i) / R;
        if (c != cprev) {
            if (cprev != 0xFFFFFFFFu) {
                ca.pop_front(1);
                ci.pop_front(1);
            }
            ca.wait_front(1);
            ci.wait_front(1);
            cprev = c;
        }
        cx.wait_front(1);
        co.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        copy_tile(cb_x, 0, 1);
        copy_init(cb_a);
        copy_tile(cb_a, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        sin_tile_init();
        sin_tile(1);
        square_tile_init();
        square_tile(1);
        copy_init(cb_i);
        copy_tile(cb_i, 0, 2);
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float32>(0, 1, 2, 0, 0x3F800000u);
        typecast_tile_init<F32, BF16>();
        typecast_tile<F32, BF16>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cx.pop_front(1);
        co.push_back(1);
    }
    if (cprev != 0xFFFFFFFFu) {
        ca.pop_front(1);
        ci.pop_front(1);
    }
}
