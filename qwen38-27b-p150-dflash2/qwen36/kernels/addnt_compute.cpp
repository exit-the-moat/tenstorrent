// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: TP decode residual add + RMSNorm in one program (kernels/addnt_reader.cpp), compute of core k on its PER tiles
// (one tile row: up to 32 users, row r = user r):
//   H_i = X_i + D_i rounded to bf16 (cb_h, written out as the new residual, as ttnn.add rounds)
//   P = sum_i ones @ (H_i * H_i)^T (matmul with B transposed): every row of P = this core's per-user sums of squares,
//       user r in column r (fp32); the writer sends row 0 of P to row k of G on every core
//   R = transpose(G) @ ones: row r = the total over the NC cores for user r, in every column; R = rsqrt(R / dim + eps)
//   Y_i = H_i * R * W_i on the SFPU in fp32 (W_i: the norm weight, replicated over the rows), rounded to bf16

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_d = 1, cb_w = 2, cb_ones = 3, cb_h = 4, cb_sq = 5, cb_p = 6, cb_g = 7, cb_gt = 8,
                   cb_r = 9, cb_out = 16;

inline void copy_in(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, dst);
}

void kernel_main() {
    constexpr uint32_t PER = get_compile_time_arg_val(0);
    constexpr uint32_t INV_N = get_compile_time_arg_val(1);
    constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(2);  // (EPS is an LLK macro)
    constexpr uint32_t HAS_D = get_compile_time_arg_val(3);
    constexpr uint32_t DBG = get_compile_time_arg_val(4);  // bit 1: no final multiply (timing only)
    constexpr uint32_t FPU_OUT = get_compile_time_arg_val(5);  // final multiplies on the FPU (else SFPU, fp32)
    constexpr uint32_t NG = get_compile_time_arg_val(6);  // G tiles (cores / 32, rounded up)
    constexpr uint32_t cb_src = HAS_D ? cb_h : cb_x;
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    CircularBuffer x(cb_x), d(cb_d), w(cb_w), ones(cb_ones), hb(cb_h), sq(cb_sq), p(cb_p), g(cb_g), gt(cb_gt),
        r(cb_r), out(cb_out);
    x.wait_front(PER);
    if constexpr (HAS_D) {
        d.wait_front(PER);
        reconfig_data_format(cb_x, cb_d);
        pack_reconfig_data_format(cb_h);
        add_tiles_init(cb_x, cb_d);
        for (uint32_t i = 0; i < PER; i++) {
            hb.reserve_back(1);
            tile_regs_acquire();
            add_tiles(cb_x, cb_d, i, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_h);
            tile_regs_release();
            hb.push_back(1);
        }
        hb.wait_front(PER);  // never popped here: the writer reads it too
    }
    // squares (exact in fp32: bf16 x bf16)
    reconfig_data_format(cb_src, cb_src);
    pack_reconfig_data_format(cb_sq);
    mul_tiles_init(cb_src, cb_src);
    for (uint32_t i = 0; i < PER; i++) {
        sq.reserve_back(1);
        tile_regs_acquire();
        mul_tiles(cb_src, cb_src, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
        sq.push_back(1);
    }
    sq.wait_front(PER);
    ones.wait_front(1);
    // P = sum_i ones @ SQ_i^T
    p.reserve_back(1);
    reconfig_data_format<SrcOrder::Reverse>(cb_ones, cb_sq);
    pack_reconfig_data_format(cb_p);
    matmul_init(cb_ones, cb_sq, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < PER; i++) {
        matmul_tiles(cb_ones, cb_sq, 0, i, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_p);
    tile_regs_release();
    p.push_back(1);
    sq.pop_front(PER);
    // G (row k = core k's P row) -> transpose (fp32 in dest) -> GT
    g.wait_front(NG);
    reconfig_data_format_srca(cb_g);
    pack_reconfig_data_format(cb_gt);
    transpose_init(cb_g);
    for (uint32_t j = 0; j < NG; j++) {
        gt.reserve_back(1);
        tile_regs_acquire();
        transpose_tile(cb_g, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gt);
        tile_regs_release();
        gt.push_back(1);
    }
    g.pop_front(NG);
    gt.wait_front(NG);
    // R = rsqrt(GT @ ones / dim + eps)
    r.reserve_back(1);
    reconfig_data_format<SrcOrder::Reverse>(cb_gt, cb_ones);
    pack_reconfig_data_format(cb_r);
    matmul_init(cb_gt, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t j = 0; j < NG; j++) {
        matmul_tiles(cb_gt, cb_ones, j, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_N);
    add_unary_tile(0, NORM_EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    r.push_back(1);
    gt.pop_front(NG);
    r.wait_front(1);
    w.wait_front(PER);
    pack_reconfig_data_format(cb_out);
    for (uint32_t i = 0; i < PER; i++) {
        out.reserve_back(1);
        tile_regs_acquire();
        if constexpr (FPU_OUT) {  // Y = (H * R) * W on the FPU: H * R in fp32 dest, then dest (as srcA) * W
            reconfig_data_format(cb_src, cb_r);
            mul_tiles_init(cb_src, cb_r);
            mul_tiles(cb_src, cb_r, i, 0, 0);
            reconfig_data_format_srcb(cb_r, cb_w);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, i, 0);
        } else {
            copy_in(cb_src, i, 0);
            if constexpr (!(DBG & 2)) {
                copy_in(cb_r, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                copy_in(cb_w, i, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
    x.pop_front(PER);
    if constexpr (HAS_D) {
        d.pop_front(PER);
    }
    w.pop_front(PER);
    ones.pop_front(1);
    r.pop_front(1);
}
