// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2: fused one-token GDN decode (TP qwen36, kernels/gdnq_reader.cpp), compute. Replaces the ~45 ttnn ops between
// the qkvzab in-projection and the out projection of TPGatedDeltaNet.forward_decode, per (sequence r, value head hv):
//   conv   y = silu(t0 st1 + t1 st2 + t2 st3 + t3 x)                       (12 channel rows packed into one tile)
//   q = l2norm(y_q) * K^-0.5, k = l2norm(y_k), v = y_v
//   decay = exp(A_neg * softplus(a + dt_bias)), beta = sigmoid(b)          (full tiles, every element the scalar)
//   u = beta (v - decay k S);  S' = decay S + k^T u  (fp32 state, SFPU in fp32, rank-1 update on the FPU)
//   o = q S' = decay q S + (q . k) u
//   out = rms_norm(o) * w_norm * silu(z)
// The state is read twice: cb_smm (matmul operand, tf32 in the FPU like the ttnn matmuls) and cb_sdst (unpacked
// straight to the fp32 DST for the decay multiply and the sum). DST is fp32 (fp32_dest_acc_en), 4 tiles.

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
constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_onw = 9,
                   cb_sel = 11, cb_ypk = 15, cb_out = 16, cb_y = 19;
// fp32
constexpr uint32_t cb_sdst = 8, cb_smm = 13, cb_qs = 14, cb_state = 17, cb_qk = 18, cb_scal = 20, cb_qn = 21,
                   cb_kn = 22, cb_kc = 23, cb_on = 24, cb_ks = 25, cb_u = 26, cb_o = 28, cb_tmp = 29, cb_decay = 30,
                   cb_beta = 31;

// copy tile `tile` of cb into DST[dst] (inside tile_regs_acquire); reconfigures srcA for cb's format every time
inline void copy_in(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, dst);
}

// o[1, N] = A[1, K] @ (tr ? B[N, K]^T : B[K, N]); A tiles from a_off, B tiles from b_off. Inputs waited.
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

// scal[0,0] = rsqrt(s * pre + eps) * post, s = [0,0] of the one tile in cb_tmp (popped)
void scalar_rsqrt(uint32_t pre_bits, uint32_t eps_bits, uint32_t post_bits) {
    CircularBuffer t(cb_tmp), s(cb_scal);
    t.wait_front(1);
    s.reserve_back(1);
    pack_reconfig_data_format(cb_scal);
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
    pack_tile(0, cb_scal);
    tile_regs_release();
    s.push_back(1);
    t.pop_front(1);
}

// o = A[a_off ..] * scalar ([0,0] of the one tile in scal), n tiles
void bcast_scalar_mul(uint32_t a, uint32_t a_off, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer co(o);
    co.reserve_back(n);
    reconfig_data_format(a, scal);
    pack_reconfig_data_format(o);
    mul_tiles_bcast_scalar_init_short(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, a_off + i, 0, 0);
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

}  // namespace

void kernel_main() {
    constexpr uint32_t QSCALE = get_compile_time_arg_val(0);        // K^-0.5 (fp32 bits)
    constexpr uint32_t L2_EPS = get_compile_time_arg_val(1);        // 1e-6
    constexpr uint32_t ONE = get_compile_time_arg_val(2);           // 1.0
    constexpr uint32_t INV_V = get_compile_time_arg_val(3);         // 1 / 128
    constexpr uint32_t NORM_EPS = get_compile_time_arg_val(4);      // the o rms_norm epsilon
    constexpr uint32_t SP_THRESHOLD = get_compile_time_arg_val(5);  // softplus threshold (20)
    const uint32_t ns = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_q, cb_w, cb_ypk);
    CircularBuffer x(cb_x), q(cb_q), w(cb_w), z(cb_z), a(cb_a), b(cb_b), dtb(cb_dtb), aneg(cb_aneg), onw(cb_onw),
        sel(cb_sel), ypk(cb_ypk), out(cb_out), y(cb_y), sdst(cb_sdst), smm(cb_smm), qs(cb_qs), st(cb_state), qk(cb_qk),
        scal(cb_scal), qn(cb_qn), kn(cb_kn), kc(cb_kc), on(cb_on), ks(cb_ks), u(cb_u), o(cb_o), decay(cb_decay),
        beta(cb_beta);

#ifdef GDNQ_TS
    cb_wait_front(27, 1);
    const uint32_t tsa = get_tile_address(27, 0);
#define CTS(k) PACK((reinterpret_cast<volatile uint32_t*>(tsa)[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)))
    CTS(16);
#else
#define CTS(k)
#endif
#ifdef GDNQ_NOOP
    // ablation: no math, only the CB handshakes (data movement alone)
    w.wait_front(4);
    sel.wait_front(12);
    dtb.wait_front(1);
    aneg.wait_front(1);
    onw.wait_front(4);
    for (uint32_t s = 0; s < ns; s++) {
        x.wait_front(1);
        q.wait_front(3);
        a.wait_front(1);
        b.wait_front(1);
        z.wait_front(4);
        smm.wait_front(16);
        sdst.wait_front(16);
        st.reserve_back(16);
        st.push_back(16);
        out.reserve_back(4);
        out.push_back(4);
        x.pop_front(1);
        q.pop_front(3);
        a.pop_front(1);
        b.pop_front(1);
        z.pop_front(4);
        smm.pop_front(16);
        sdst.pop_front(16);
    }
    w.pop_front(4);
    sel.pop_front(12);
    dtb.pop_front(1);
    aneg.pop_front(1);
    onw.pop_front(4);
    return;
#endif
    w.wait_front(4);
    for (uint32_t s = 0; s < ns; s++) {
        // ---- conv on the packed tiles
        x.wait_front(1);
        q.wait_front(3);
        ypk.reserve_back(1);
        pack_reconfig_data_format(cb_ypk);
        reconfig_data_format(cb_q, cb_w);
        mul_tiles_init(cb_q, cb_w);
        tile_regs_acquire();
        mul_tiles(cb_q, cb_w, 0, 0, 1);
        mul_tiles(cb_q, cb_w, 1, 1, 2);
        mul_tiles(cb_q, cb_w, 2, 2, 3);
        reconfig_data_format(cb_x, cb_w);
        mul_tiles_init(cb_x, cb_w);
        mul_tiles(cb_x, cb_w, 0, 3, 0);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_ypk);
        tile_regs_release();
        ypk.push_back(1);
        x.pop_front(1);
        CTS(17 + 12 * s);
        q.pop_front(3);

        // ---- unpack: y_t = E_t @ ypk moves packed row t to row 0
        if (s == 0) {
            sel.wait_front(12);
        }
        ypk.wait_front(1);
        y.reserve_back(12);
        reconfig_data_format<SrcOrder::Reverse>(cb_sel, cb_ypk);
        pack_reconfig_data_format(cb_y);
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
        CTS(18 + 12 * s);
        y.wait_front(12);

        // ---- q / k l2 norms, k as a column
        mm(cb_y, 0, cb_y, 0, cb_tmp, 4, 1, true);
        scalar_rsqrt(ONE, L2_EPS, QSCALE);
        scal.wait_front(1);
        bcast_scalar_mul(cb_y, 0, cb_scal, cb_qn, 4);
        scal.pop_front(1);
        mm(cb_y, 4, cb_y, 4, cb_tmp, 4, 1, true);
        scalar_rsqrt(ONE, L2_EPS, ONE);
        scal.wait_front(1);
        bcast_scalar_mul(cb_y, 4, cb_scal, cb_kn, 4);
        scal.pop_front(1);
        qn.wait_front(4);
        kn.wait_front(4);
        transpose_block(cb_kn, cb_kc, 4);
        kc.wait_front(4);
        CTS(19 + 12 * s);

        // ---- gates
        a.wait_front(1);
        b.wait_front(1);
        if (s == 0) {
            dtb.wait_front(1);
            aneg.wait_front(1);
        }
        decay.reserve_back(1);
        pack_reconfig_data_format(cb_decay);
        reconfig_data_format(cb_a, cb_dtb);
        add_tiles_init(cb_a, cb_dtb);
        tile_regs_acquire();
        add_tiles(cb_a, cb_dtb, 0, 0, 0);
        softplus_tile_init();
        softplus_tile(0, ONE, ONE, SP_THRESHOLD);
        copy_in(cb_aneg, 0, 1);
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
        pack_reconfig_data_format(cb_beta);
        tile_regs_acquire();
        copy_in(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        beta.push_back(1);
        a.pop_front(1);
        CTS(20 + 12 * s);
        b.pop_front(1);
        decay.wait_front(1);
        beta.wait_front(1);

        // ---- k S, q S, q . k
        smm.wait_front(16);
        mm(cb_kn, 0, cb_smm, 0, cb_ks, 4, 4, false);
        mm(cb_qn, 0, cb_smm, 0, cb_qs, 4, 4, false);
        smm.pop_front(16);
        mm(cb_qn, 0, cb_kc, 0, cb_qk, 4, 1, false);
        ks.wait_front(4);
        qs.wait_front(4);
        qk.wait_front(1);
        CTS(21 + 12 * s);

        // ---- u = (v - decay k S) * beta
        u.reserve_back(4);
        pack_reconfig_data_format(cb_u);
        for (uint32_t j = 0; j < 4; j++) {
            tile_regs_acquire();
            copy_in(cb_y, 8 + j, 0);
            copy_in(cb_ks, j, 1);
            copy_in(cb_decay, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            copy_in(cb_beta, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_u);
            tile_regs_release();
        }
        u.push_back(4);
        ks.pop_front(4);
        CTS(22 + 12 * s);
        u.wait_front(4);

        // ---- S' = decay S + kcol (x) u -> cb_state (the writer stores it)
        sdst.wait_front(16);
        st.reserve_back(16);
        pack_reconfig_data_format(cb_state);
        for (uint32_t mi = 0; mi < 4; mi++) {
            for (uint32_t j = 0; j < 4; j += 2) {
                tile_regs_acquire();
                copy_in(cb_sdst, mi * 4 + j, 0);
                copy_in(cb_sdst, mi * 4 + j + 1, 1);
                copy_in(cb_decay, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                mul_binary_tile(1, 2, 1);
                reconfig_data_format<SrcOrder::Reverse>(cb_kc, cb_u);
                matmul_init(cb_kc, cb_u, 0);
                matmul_tiles(cb_kc, cb_u, mi, j, 0);
                matmul_tiles(cb_kc, cb_u, mi, j + 1, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_state, mi * 4 + j);
                pack_tile<true>(1, cb_state, mi * 4 + j + 1);
                tile_regs_release();
            }
        }
        st.push_back(16);
        sdst.pop_front(16);
        CTS(23 + 12 * s);

        // ---- o = decay q S + (q . k) u
        o.reserve_back(4);
        pack_reconfig_data_format(cb_o);
        for (uint32_t j = 0; j < 4; j++) {
            tile_regs_acquire();
            reconfig_data_format(cb_u, cb_qk);
            mul_tiles_bcast_scalar_init_short(cb_u, cb_qk);
            mul_tiles_bcast_scalar(cb_u, cb_qk, j, 0, 0);
            copy_in(cb_qs, j, 1);
            copy_in(cb_decay, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_o);
            tile_regs_release();
        }
        o.push_back(4);
        u.pop_front(4);
        CTS(24 + 12 * s);
        qs.pop_front(4);
        qk.pop_front(1);
        decay.pop_front(1);
        beta.pop_front(1);
        qn.pop_front(4);
        kn.pop_front(4);
        kc.pop_front(4);

        // ---- out = rms_norm(o) * w_norm * silu(z)
        o.wait_front(4);
        mm(cb_o, 0, cb_o, 0, cb_tmp, 4, 1, true);
        scalar_rsqrt(INV_V, NORM_EPS, ONE);
        scal.wait_front(1);
        bcast_scalar_mul(cb_o, 0, cb_scal, cb_on, 4);
        scal.pop_front(1);
        o.pop_front(4);
        on.wait_front(4);
        z.wait_front(4);
        CTS(25 + 12 * s);
        if (s == 0) {
            onw.wait_front(4);
        }
        out.reserve_back(4);
        pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < 4; i++) {
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
        CTS(26 + 12 * s);
        out.push_back(4);
        on.pop_front(4);
        z.pop_front(4);
        y.pop_front(12);
    }
    w.pop_front(4);
    sel.pop_front(12);
    dtb.pop_front(1);
    aneg.pop_front(1);
    onw.pop_front(4);
}
