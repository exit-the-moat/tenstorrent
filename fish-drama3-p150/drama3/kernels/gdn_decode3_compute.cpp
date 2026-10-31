// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port5: fused one-token GDN decode v3 (one core per value head: kernels/gdn_decode3_reader.cpp), compute. Replaces the ~50 ttnn ops between the mega projection
// [qkv | a | b | z] and o_proj (Drama 3 single-device decode: partial-sum conv, QSCALE lean recurrence, no-clip
// gate) for one key head hk = value heads 2hk, 2hk+1:
//   conv  y = silu(Q0 + w3 x); Q0' = Q1 + w2 x; Q1' = Q2 + w1 x; Q2' = w0 x           (16 channel tiles)
//   q = l2norm(y_q) * K^-0.5, k = l2norm(y_k)                                       (sum of squares as q @ q^T)
//   per value head j: decay = exp(A_neg * softplus(a + dt_bias)), beta = sigmoid(b)
//     sd = S * decay; u = beta * (v - k . sd); S' = sd + k^T (x) u; o = q . S'      (the upstream fused recurrence)
//     o = rms_norm(o) * w_norm * silu(z)
// Every CB is bf16, DST is fp32 (fp32_dest_acc_en). Row 0 of each [1, D] tile carries the data, rows 1..31 are 0.
// The recurrence helpers follow the compute kernel of ttnn fused_recurrent_gated_delta_rule.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_gate = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_s = 8,
                   cb_onw = 9, cb_sel = 11, cb_ypk = 15;
constexpr uint32_t cb_out = 16, cb_state = 17, cb_qn = 18, cb_y = 19, cb_scal = 20, cb_qnorm = 21, cb_knorm = 22,
                   cb_kcol = 23, cb_sd = 24, cb_vread = 25, cb_u = 26, cb_snew = 27, cb_o = 28, cb_tmp = 29,
                   cb_decay = 30, cb_beta = 31;
constexpr uint32_t DST_TILES = 4;
constexpr uint32_t Kt = 4, Vt = 4, KV = 16;

// out[1, N] = A[1, K] @ (tr ? B[N, K]^T : B[K, N]); A tiles from a_off, B tiles from b_off. Inputs waited.
void mm(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o, uint32_t K, uint32_t N, bool tr) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(N);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t ni = 0; ni < N; ni++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < K; ki++) {
            matmul_tiles(a, b, a_off + ki, b_off + (tr ? ni * K + ki : ki * N + ni), 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    cb_o_.push_back(N);
}

// scal[0,0] = rsqrt(s * pre + eps) * post, s = the [0,0] element of the one tile in cb_tmp (popped)
void scalar_rsqrt(uint32_t pre_bits, uint32_t eps_bits, uint32_t post_bits) {
    CircularBuffer t(cb_tmp), s(cb_scal);
    t.wait_front(1);
    s.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_tmp);
    copy_tile(cb_tmp, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, pre_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scal);
    tile_regs_release();
    s.push_back(1);
    t.pop_front(1);
}

// out = A[a_off ..] * scalar ([0,0] of the one tile in scal), n tiles
void bcast_scalar_mul(uint32_t a, uint32_t a_off, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, a_off + i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

// u = (A[a_off ..] - B) * beta[0,0], n tiles (FPU sub, SFPU column-broadcast multiply; beta sits at [0,0])
void delta_rule_residual(uint32_t a, uint32_t a_off, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, a_off + i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

// S' = sd + kcol (x) u, packed to snew and to the state output
void rank1_update() {
    CircularBuffer cb_n(cb_snew), cb_st(cb_state);
    cb_n.reserve_back(KV);
    cb_st.reserve_back(KV);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        tile_regs_acquire();
        copy_init(cb_sd);
        for (uint32_t j = 0; j < Vt; j++) {
            copy_tile(cb_sd, mi * Vt + j, j);
        }
        matmul_init(cb_kcol, cb_u, 0);
        for (uint32_t j = 0; j < Vt; j++) {
            matmul_tiles(cb_kcol, cb_u, mi, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Vt; j++) {
            pack_tile<true>(j, cb_snew, mi * Vt + j);
            pack_tile<true>(j, cb_state, mi * Vt + j);
        }
        tile_regs_release();
    }
    cb_n.push_back(KV);
    cb_st.push_back(KV);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t QSCALE = get_compile_time_arg_val(0);    // K^-0.5 (fp32 bits): q's extra scale
    constexpr uint32_t L2_EPS = get_compile_time_arg_val(1);    // 1e-6
    constexpr uint32_t ONE = get_compile_time_arg_val(2);       // 1.0
    constexpr uint32_t INV_V = get_compile_time_arg_val(3);     // 1 / 128
    constexpr uint32_t NORM_EPS = get_compile_time_arg_val(4);  // the o rms_norm epsilon
    constexpr uint32_t SP_THRESHOLD = get_compile_time_arg_val(5);  // softplus threshold (20)

    compute_kernel_hw_startup(cb_x, cb_w, cb_y);
    CircularBuffer x(cb_x), q(cb_q), w(cb_w), y(cb_y), qn(cb_qn), gate(cb_gate), a(cb_a), b(cb_b), dtb(cb_dtb),
        aneg(cb_aneg), s(cb_s), onw(cb_onw), qnorm(cb_qnorm), knorm(cb_knorm), kcol(cb_kcol), sd(cb_sd),
        vread(cb_vread), u(cb_u), snew(cb_snew), o(cb_o), tmp(cb_tmp), decay(cb_decay), beta(cb_beta), scal(cb_scal);

    {
        DeviceZoneScopedN("G_CONV");
    // ---- conv on the packed tiles (row t = channel tile t): y = silu(Q0 + w3 x), the new partial sums
    {
        DeviceZoneScopedN("G_CWAIT");
        x.wait_front(1);
        q.wait_front(3);
        w.wait_front(4);
    }
    CircularBuffer ypk(cb_ypk), sel(cb_sel);
    ypk.reserve_back(1);
    qn.reserve_back(3);
    // products and partial sums meet in fp32 DST and are added on the SFPU (an FPU dest-reuse add would truncate the
    // fp32 product on its way into a source register): y = silu(Q0 + w3 x), Q0' = Q1 + w2 x
    tile_regs_acquire();
    mul_init(cb_x, cb_w, false);
    mul_tiles(cb_x, cb_w, 0, 3, 0);
    mul_tiles(cb_x, cb_w, 0, 2, 2);
    copy_init(cb_q);
    copy_tile(cb_q, 0, 1);
    copy_tile(cb_q, 1, 3);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(2, 3, 2);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_ypk);
    pack_tile(2, cb_qn);
    tile_regs_release();
    // Q1' = Q2 + w1 x, Q2' = w0 x
    tile_regs_acquire();
    mul_init(cb_x, cb_w, false);
    mul_tiles(cb_x, cb_w, 0, 1, 0);
    mul_tiles(cb_x, cb_w, 0, 0, 2);
    copy_init(cb_q);
    copy_tile(cb_q, 2, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_qn);
    pack_tile(2, cb_qn);
    tile_regs_release();
    ypk.push_back(1);
    qn.push_back(3);
    x.pop_front(1);
    q.pop_front(3);
    w.pop_front(4);
    // unpack: y_t = E_t @ ypk puts packed row t into row 0 (E_t: one 1 at [0, t]); rows 1..31 come out 0
    ypk.wait_front(1);
    {
        DeviceZoneScopedN("G_SWAIT");
        sel.wait_front(12);
    }
    {
    DeviceZoneScopedN("G_UNPACK");
    y.reserve_back(12);
    matmul_init(cb_sel, cb_ypk, 0);
    for (uint32_t t = 0; t < 12; t++) {
        tile_regs_acquire();
        matmul_tiles(cb_sel, cb_ypk, t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
    }
    y.push_back(12);
    ypk.pop_front(1);
    sel.pop_front(12);
    }
    }
    y.wait_front(12);

    {
        DeviceZoneScopedN("G_QKNORM");
    // ---- q = y_q / sqrt(sum y_q^2 + eps) * K^-0.5, k = y_k / sqrt(sum y_k^2 + eps)
    mm(cb_y, 0, cb_y, 0, cb_tmp, Kt, 1, true);
    scalar_rsqrt(ONE, L2_EPS, QSCALE);
    scal.wait_front(1);
    bcast_scalar_mul(cb_y, 0, cb_scal, cb_qnorm, Kt);
    scal.pop_front(1);
    mm(cb_y, Kt, cb_y, Kt, cb_tmp, Kt, 1, true);
    scalar_rsqrt(ONE, L2_EPS, ONE);
    scal.wait_front(1);
    bcast_scalar_mul(cb_y, Kt, cb_scal, cb_knorm, Kt);
    scal.pop_front(1);
    qnorm.wait_front(Kt);
    knorm.wait_front(Kt);
    transpose_block(cb_knorm, cb_kcol, Kt);
    kcol.wait_front(Kt);
    }
    gate.wait_front(4);
    onw.wait_front(4);

    for (uint32_t j = 0; j < 1; j++) {  // one value head per core (v3)
        {
            DeviceZoneScopedN("G_GATES");
        // ---- gates: decay = exp(A_neg * softplus(a + dt_bias)), beta = sigmoid(b) (at [0,0])
        a.wait_front(1);
        b.wait_front(1);
        dtb.wait_front(1);
        aneg.wait_front(1);
        decay.reserve_back(1);
        tile_regs_acquire();
        add_init(cb_a, cb_dtb);
        add_tiles(cb_a, cb_dtb, 0, 0, 0);
        softplus_tile_init();
        softplus_tile(0, ONE, ONE, SP_THRESHOLD);
        copy_init(cb_aneg);
        copy_tile(cb_aneg, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_decay);
        tile_regs_release();
        decay.push_back(1);
        beta.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_b);
        copy_tile(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        beta.push_back(1);
        a.pop_front(1);
        b.pop_front(1);
        dtb.pop_front(1);
        aneg.pop_front(1);

        }
        {
            DeviceZoneScopedN("G_REC");
        // ---- recurrence
        s.wait_front(KV);
        decay.wait_front(1);
        bcast_scalar_mul(cb_s, 0, cb_decay, cb_sd, KV);
        s.pop_front(KV);
        decay.pop_front(1);
        sd.wait_front(KV);
        mm(cb_knorm, 0, cb_sd, 0, cb_vread, Kt, Vt, false);
        vread.wait_front(Vt);
        beta.wait_front(1);
        delta_rule_residual(cb_y, 2 * Kt + j * Vt, cb_vread, cb_beta, cb_u, Vt);
        vread.pop_front(Vt);
        beta.pop_front(1);
        u.wait_front(Vt);
        rank1_update();
        u.pop_front(Vt);
        sd.pop_front(KV);
        snew.wait_front(KV);
        mm(cb_qnorm, 0, cb_snew, 0, cb_o, Kt, Vt, false);
        snew.pop_front(KV);
        o.wait_front(Vt);

        }
        {
            DeviceZoneScopedN("G_ONORM");
        // ---- o = rms_norm(o) * w_norm * silu(z)
        mm(cb_o, 0, cb_o, 0, cb_tmp, Vt, 1, true);
        scalar_rsqrt(INV_V, NORM_EPS, ONE);
        scal.wait_front(1);
        bcast_scalar_mul(cb_o, 0, cb_scal, cb_tmp, Vt);
        scal.pop_front(1);
        o.pop_front(Vt);
        tmp.wait_front(Vt);
        o.reserve_back(Vt);
        mul_init(cb_tmp, cb_onw, false);
        for (uint32_t i = 0; i < Vt; i++) {
            tile_regs_acquire();
            mul_tiles(cb_tmp, cb_onw, i, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_o);
            tile_regs_release();
        }
        o.push_back(Vt);
        tmp.pop_front(Vt);
        o.wait_front(Vt);
        CircularBuffer out(cb_out);
        out.reserve_back(Vt);
        for (uint32_t i = 0; i < Vt; i++) {
            tile_regs_acquire();
            copy_init(cb_gate);
            copy_tile(cb_gate, j * Vt + i, 0);
            silu_tile_init();
            silu_tile(0);
            copy_init(cb_o);
            copy_tile(cb_o, i, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        out.push_back(Vt);
        o.pop_front(Vt);
        }
    }
    qnorm.pop_front(Kt);
    knorm.pop_front(Kt);
    kcol.pop_front(Kt);
    gate.pop_front(4);
    onw.pop_front(4);
    y.pop_front(12);
}
