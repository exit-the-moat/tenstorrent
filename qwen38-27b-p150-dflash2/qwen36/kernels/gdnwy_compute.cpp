// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ms4 (Exit the Moat, Episode 2): the DFlash2 verify GDN chain in block (WY) form, compute. Rows of every tile = the NT
// slots of this stream (rows NT .. 31 exactly 0 where it matters). Per value head (own state columns c0 .. c0+NC-1):
//   S0   = gamma_a * base + K_prev^T (W (.) U_prev)                     (W: row j = R_prev[a, j]; a >= NT: S0 = base)
//   g    = A_neg * softplus(a + dt_bias)   (log decay per slot); the reader sums it in fp32: Gc_t, D[t, j] = Gc_t - Gc_j
//   gam  = exp(Gc), R = exp(D) (.) Mi                                     (R[t, j] = gamma_t / gamma_j, j <= t)
//   L    = beta (.) (K K^T) (.) R (.) Ms,  inv = (I - L)(I + L^2)(I + L^4) = (I + L)^-1   (L^8 = 0)
//   U    = inv @ (beta (.) (V - gam (.) K S0))
//   O    = gam (.) Q S0 + ((Q K^T) (.) R) @ U   -> o_all, then rms norm * w * silu(z) of all slots (gdnq4c RMSALL)
// and K / U / R / gam go to blk for the next step. The same recurrence as the NT chained slots (gdnq4c), summed in a
// different order (a numerics change: judged gate). CPU check of the algebra: ms4 wy_check.py (fp64 exact).
// GDNWY_COMMIT: S0 only.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// bf16
constexpr uint32_t cb_z = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_onw = 9, cb_out = 16;
// fp32
constexpr uint32_t cb_sdst = 8, cb_qs = 14, cb_state = 17, cb_on = 24, cb_ks = 25, cb_u = 26, cb_o = 28, cb_tmp = 29,
                   cb_g = 35, cb_beta = 36, cb_qn = 42, cb_kn = 43, cb_oall = 46, cb_rm = 47, cb_gam = 48, cb_kp = 50,
                   cb_up = 51, cb_wr = 52, cb_gu = 53, cb_kt = 54, cb_uw = 55, cb_gc = 56, cb_bm = 57, cb_l = 58,
                   cb_inv = 59, cb_rhs = 60, cb_mq = 61, cb_cel = 62, cb_gct = 63, cb_a1 = 0, cb_l2 = 1, cb_l2b = 2,
                   cb_l4 = 11, cb_p1 = 12, cb_p2 = 13, cb_p3 = 15, cb_t2 = 18, cb_qk = 19, cb_ld = 20, cb_s0 = 21;
constexpr uint32_t cb_orecv = 33;
// constant tiles: cb_cel 0 = Ms, 1 = Mi, 2 = RowMask, 3 = I

inline void copy_in(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, dst);
}

void mm(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o, uint32_t K, uint32_t N, bool tr) {
    CircularBuffer co(o);
    co.reserve_back(N);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    pack_reconfig_data_format(o);
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
    co.push_back(N);
}

// one-tile product a @ b (K = 1) packed into o1 and o2
void mm1x2(uint32_t a, uint32_t b, uint32_t o1, uint32_t o2) {
    CircularBuffer c1(o1), c2(o2);
    c1.reserve_back(1);
    c2.reserve_back(1);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b, 0);
    tile_regs_acquire();
    matmul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(o1);
    pack_tile(0, o1);
    pack_reconfig_data_format(o2);
    pack_tile(0, o2);
    tile_regs_release();
    c1.push_back(1);
    c2.push_back(1);
}

void tile_rsqrt_to(uint32_t pre_bits, uint32_t eps_bits, uint32_t post_bits, uint32_t o) {
    CircularBuffer t(cb_tmp), s(o);
    t.wait_front(1);
    s.reserve_back(1);
    pack_reconfig_data_format(o);
    tile_regs_acquire();
    copy_in(cb_tmp, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, pre_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
    s.push_back(1);
    t.pop_front(1);
}

// out tiles o[0..n) = a[a_off + i] (.) b[b_off] (FPU, the b tile carries its row / column pattern itself)
void wy_mul(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o, uint32_t n) {
    CircularBuffer co(o);
    co.reserve_back(n);
    reconfig_data_format(a, b);
    pack_reconfig_data_format(o);
    mul_tiles_init(a, b);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(a, b, a_off + i, b_off, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    co.push_back(n);
}

void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer co(o);
    co.reserve_back(n);
    reconfig_data_format_srca(in);
    pack_reconfig_data_format(o);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    co.push_back(n);
}

// o = x (+/-) I  (SFPU, exact fp32; x from an unpack-to-dest CB)
void eye_pm(uint32_t x, uint32_t o, bool minus) {
    CircularBuffer co(o);
    co.reserve_back(1);
    pack_reconfig_data_format(o);
    tile_regs_acquire();
    copy_in(cb_cel, 3, 0);
    copy_in(x, 0, 1);
    if (minus) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
    } else {
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
    co.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t QSCALE = get_compile_time_arg_val(0);
    constexpr uint32_t L2_EPS = get_compile_time_arg_val(1);
    constexpr uint32_t ONE = get_compile_time_arg_val(2);
    constexpr uint32_t INV_V = get_compile_time_arg_val(3);
    constexpr uint32_t NORM_EPS = get_compile_time_arg_val(4);
    constexpr uint32_t SP_THRESHOLD = get_compile_time_arg_val(5);
    constexpr uint32_t CS = get_compile_time_arg_val(6);
    constexpr uint32_t NC = 4 / CS;
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    (void)QSCALE;
    (void)L2_EPS;
    (void)INV_V;
    (void)NORM_EPS;
    (void)SP_THRESHOLD;
    (void)c0;

    compute_kernel_hw_startup(cb_kp, cb_up, cb_state);

    // ---- S0 = gamma_a * base + K_prev^T (W (.) U_prev)  -> cb_state (own columns, tiles mi * NC + j)
    {
        CircularBuffer kp(cb_kp), up(cb_up), wr(cb_wr), gu(cb_gu), sd(cb_sdst), st(cb_state), kt(cb_kt), uw(cb_uw);
        kp.wait_front(4);
        up.wait_front(NC);
        wr.wait_front(1);
        gu.wait_front(1);
        transpose_block(cb_kp, cb_kt, 4);
        wy_mul(cb_up, 0, cb_wr, 0, cb_uw, NC);
        kt.wait_front(4);
        uw.wait_front(NC);
        sd.wait_front(4 * NC);
        st.reserve_back(4 * NC);
#ifndef GDNWY_COMMIT
        CircularBuffer(cb_s0).reserve_back(4 * NC);
#endif
        for (uint32_t mi = 0; mi < 4; mi++) {
            for (uint32_t j = 0; j < NC; j++) {
                tile_regs_acquire();
                copy_in(cb_sdst, mi * NC + j, 0);
                copy_in(cb_gu, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                reconfig_data_format<SrcOrder::Reverse>(cb_kt, cb_uw);
                matmul_init(cb_kt, cb_uw, 0);
                matmul_tiles(cb_kt, cb_uw, mi, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_state);
                pack_tile<true>(0, cb_state, mi * NC + j);
#ifndef GDNWY_COMMIT
                pack_tile<true>(0, cb_s0, mi * NC + j);
#endif
                tile_regs_release();
            }
        }
        st.push_back(4 * NC);
#ifndef GDNWY_COMMIT
        CircularBuffer(cb_s0).push_back(4 * NC);
#endif
        sd.pop_front(4 * NC);
        kt.pop_front(4);
        uw.pop_front(NC);
        kp.pop_front(4);
        up.pop_front(NC);
        wr.pop_front(1);
        gu.pop_front(1);
    }
#ifndef GDNWY_COMMIT
    CircularBuffer dtb(cb_dtb), aneg(cb_aneg), cel(cb_cel);
    dtb.wait_front(1);
    aneg.wait_front(1);
    // ---- g = A_neg * softplus(a + dt_bias) (no exp: the log decay) and beta = sigmoid(b), rows = slots
    {
        CircularBuffer a(cb_a), b(cb_b), gg(cb_g), bb(cb_beta);
        a.wait_front(1);
        b.wait_front(1);
        gg.reserve_back(1);
        pack_reconfig_data_format(cb_g);
        reconfig_data_format(cb_a, cb_dtb);
        add_tiles_init(cb_a, cb_dtb);
        tile_regs_acquire();
        add_tiles(cb_a, cb_dtb, 0, 0, 0);
        softplus_tile_init();
        softplus_tile(0, ONE, ONE, SP_THRESHOLD);
        copy_in(cb_aneg, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_g);
        tile_regs_release();
        gg.push_back(1);
        bb.reserve_back(1);
        pack_reconfig_data_format(cb_beta);
        tile_regs_acquire();
        copy_in(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        bb.push_back(1);
        a.pop_front(1);
        b.pop_front(1);
    }
    // ---- GDNQ_PRO prologue (as gdnq4c_compute.cpp): conv of every channel tile (rows = slots) -> cb 41, q / k norms
    constexpr uint32_t NT9 = 8 + NC;
    {
        CircularBuffer pa(37), pw(38), yall(41), col(40);
        yall.reserve_back(NT9);
        pack_reconfig_data_format(41);
        reconfig_data_format(37, 38);
        mul_tiles_init(37, 38);
        for (uint32_t t = 0; t < NT9; t++) {
            pa.wait_front(4);
            pw.wait_front(4);
            tile_regs_acquire();
            mul_tiles(37, 38, 0, 0, 1);
            mul_tiles(37, 38, 1, 1, 2);
            mul_tiles(37, 38, 2, 2, 3);
            mul_tiles(37, 38, 3, 3, 0);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);
            add_binary_tile(0, 3, 0);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 41);
            tile_regs_release();
            pa.pop_front(4);
            pw.pop_front(4);
            mul_tiles_init(37, 38);
        }
        yall.push_back(NT9);
        yall.wait_front(NT9);
        mm(41, 0, 41, 0, cb_tmp, 4, 1, true);
        tile_rsqrt_to(ONE, L2_EPS, QSCALE, 39);
        col.wait_front(1);
        wy_mul(41, 0, 40, 0, cb_qn, 4);
        col.pop_front(1);
        mm(41, 4, 41, 4, cb_tmp, 4, 1, true);
        tile_rsqrt_to(ONE, L2_EPS, ONE, 39);
        col.wait_front(1);
        wy_mul(41, 4, 40, 0, cb_kn, 4);
        col.pop_front(1);
    }
    CircularBuffer qn(cb_qn), kn(cb_kn), yall(41);
    qn.wait_front(4);
    kn.wait_front(4);
    cel.wait_front(4);
    // ---- gam = exp(Gc), R = exp(D) (.) Mi from the reader fp32 sums (Gfull: row t = Gc_t; D[t, j] = Gc_t - Gc_j,
    // j <= t, 0 elsewhere), betam = beta rows (reader, rows >= NT zero)
    CircularBuffer gfu(cb_gc), dd(cb_gct), gm(cb_gam), rm(cb_rm), bm(cb_bm);
    gfu.wait_front(1);
    dd.wait_front(1);
    gm.reserve_back(1);
    pack_reconfig_data_format(cb_gam);
    tile_regs_acquire();
    copy_in(cb_gc, 0, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_gam);
    tile_regs_release();
    gm.push_back(1);
    rm.reserve_back(1);
    pack_reconfig_data_format(cb_rm);
    tile_regs_acquire();
    copy_in(cb_gct, 0, 0);
    exp_tile_init();
    exp_tile(0);
    copy_in(cb_cel, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rm);
    tile_regs_release();
    rm.push_back(1);
    gfu.pop_front(1);
    dd.pop_front(1);
    gm.wait_front(1);
    rm.wait_front(1);
    bm.wait_front(1);
    // ---- L = betam (.) (K K^T) (.) R (.) Ms -> cb_l (matmul operand) and cb_ld (unpack-to-dest copy for I - L)
    {
        CircularBuffer a1(cb_a1), l(cb_l), ld(cb_ld);
        mm(cb_kn, 0, cb_kn, 0, cb_a1, 4, 1, true);
        a1.wait_front(1);
        l.reserve_back(1);
        ld.reserve_back(1);
        tile_regs_acquire();
        copy_in(cb_a1, 0, 0);
        copy_in(cb_bm, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_in(cb_rm, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_in(cb_cel, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_l);
        pack_tile(0, cb_l);
        pack_reconfig_data_format(cb_ld);
        pack_tile(0, cb_ld);
        tile_regs_release();
        l.push_back(1);
        ld.push_back(1);
        a1.pop_front(1);
    }
    // ---- inv = (I - L)(I + L^2)(I + L^4)
    {
        CircularBuffer l(cb_l), ld(cb_ld), l2(cb_l2), l2b(cb_l2b), l4(cb_l4), p1(cb_p1), p2(cb_p2), p3(cb_p3), t2(cb_t2);
        l.wait_front(1);
        ld.wait_front(1);
        mm1x2(cb_l, cb_l, cb_l2, cb_l2b);  // L^2: cb_l2 (dest copy), cb_l2b (matmul operand)
        l2.wait_front(1);
        l2b.wait_front(1);
        mm(cb_l2b, 0, cb_l2b, 0, cb_l4, 1, 1, false);
        l4.wait_front(1);
        eye_pm(cb_ld, cb_p1, true);
        eye_pm(cb_l2, cb_p2, false);
        eye_pm(cb_l4, cb_p3, false);
        p1.wait_front(1);
        p2.wait_front(1);
        p3.wait_front(1);
        mm(cb_p1, 0, cb_p2, 0, cb_t2, 1, 1, false);
        t2.wait_front(1);
        mm(cb_t2, 0, cb_p3, 0, cb_inv, 1, 1, false);
        for (uint32_t c : {cb_l, cb_ld, cb_l2, cb_l2b, cb_l4, cb_p1, cb_p2, cb_p3, cb_t2}) {
            CircularBuffer(c).pop_front(1);
        }
    }
    // ---- K S0, Q S0 (own columns), U = inv @ (betam (.) (V - gam (.) K S0))
    CircularBuffer st(cb_s0), ks(cb_ks), qs(cb_qs), inv(cb_inv), rhs(cb_rhs), u(cb_u);
    st.wait_front(4 * NC);
    mm(cb_kn, 0, cb_s0, 0, cb_ks, 4, NC, false);
    mm(cb_qn, 0, cb_s0, 0, cb_qs, 4, NC, false);
    ks.wait_front(NC);
    qs.wait_front(NC);
    inv.wait_front(1);
    rhs.reserve_back(NC);
    pack_reconfig_data_format(cb_rhs);
    for (uint32_t j = 0; j < NC; j++) {
        tile_regs_acquire();
        copy_in(41, 8 + j, 0);
        copy_in(cb_ks, j, 1);
        copy_in(cb_gam, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        copy_in(cb_bm, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rhs);
        tile_regs_release();
    }
    rhs.push_back(NC);
    rhs.wait_front(NC);
    mm(cb_inv, 0, cb_rhs, 0, cb_u, 1, NC, false);
    u.wait_front(NC);
    // ---- Mq = (Q K^T) (.) R ; O = gam (.) Q S0 + Mq @ U  -> cb_o (own column tiles, to o_all by the writer)
    {
        CircularBuffer qk(cb_qk), mq(cb_mq), o(cb_o);
        mm(cb_qn, 0, cb_kn, 0, cb_qk, 4, 1, true);
        qk.wait_front(1);
        mq.reserve_back(1);
        pack_reconfig_data_format(cb_mq);
        tile_regs_acquire();
        copy_in(cb_qk, 0, 0);
        copy_in(cb_rm, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mq);
        tile_regs_release();
        mq.push_back(1);
        mq.wait_front(1);
        o.reserve_back(NC);
        for (uint32_t j = 0; j < NC; j++) {
            tile_regs_acquire();
            copy_in(cb_qs, j, 0);
            copy_in(cb_gam, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            reconfig_data_format<SrcOrder::Reverse>(cb_mq, cb_u);
            matmul_init(cb_mq, cb_u, 0);
            matmul_tiles(cb_mq, cb_u, 0, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            pack_tile(0, cb_o);
            tile_regs_release();
        }
        o.push_back(NC);
        qk.pop_front(1);
        mq.pop_front(1);
    }
    // ---- out = rms_norm(o) * w_norm * silu(z) of ALL slots (gdnq4c RMSALL; o_all from the writer)
    {
        CircularBuffer oall(cb_oall), col(40), on(cb_on), z(cb_z), onw(cb_onw), out(cb_out), orecv(cb_orecv);
        orecv.wait_front(1);
        oall.wait_front(4);
        mm(cb_oall, 0, cb_oall, 0, cb_tmp, 4, 1, true);
        tile_rsqrt_to(INV_V, NORM_EPS, ONE, 39);
        col.wait_front(1);
        on.reserve_back(NC);
        reconfig_data_format(cb_oall, 40);
        pack_reconfig_data_format(cb_on);
        mul_tiles_init(cb_oall, 40);
        for (uint32_t j = 0; j < NC; j++) {
            tile_regs_acquire();
            mul_tiles(cb_oall, 40, c0 + j, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_on);
            tile_regs_release();
        }
        on.push_back(NC);
        col.pop_front(1);
        on.wait_front(NC);
        z.wait_front(NC);
        onw.wait_front(NC);
        out.reserve_back(NC);
        pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < NC; i++) {
            tile_regs_acquire();
            copy_in(cb_z, i, 0);
            silu_tile_init();
            silu_tile(0);
            copy_in(cb_on, i, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_in(cb_onw, i, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        out.push_back(NC);
        on.pop_front(NC);
        z.pop_front(NC);
        orecv.pop_front(1);
    }
    // K / R / gam / U stay in their CBs until the writer stored them (never popped: the program ends)
    dtb.pop_front(1);
    aneg.pop_front(1);
#endif
}
