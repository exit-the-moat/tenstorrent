// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3: gdnq4_compute.cpp looped over the GDNQ_CHAIN slots of the DFlash2 verify (one sequence, ns = 1 per slot;
// see gdnq4c_reader.cpp): per slot exactly the gdnq4 instruction sequence; dt_bias / A_neg / taps / selectors / norm
// weight stay in their CBs until the last slot.
// (gdnq4:) ep2 gdn: fused one-token GDN decode v4 (QWEN36_GDN_FAST=4), compute: v3 with the gates of both sequences first
// (decay / beta CBs hold 2 tiles). v3: As gdnq2_compute.cpp, with the v3 packed
// layout (gdnq3_reader.cpp): ONE conv for both sequences of the pair (packed rows R(t, s)), the unpack of sequence s
// with its own selector tiles (s * NSEL + k). Per element the same operations as v1 / v2.
// v2 notes: the math of gdnq_compute.cpp, tile for tile, on the NC = 4 / CS state column tiles [c0, c0 + NC):
//   conv, unpack (q, k and the own v rows), q / k l2 norms, gates: as v1 (every core of the head computes them)
//   kS, qS: [1 x 4] @ S[:, own columns] (NC tiles each), q . k
//   u = (v - decay k S) * beta, o = decay q S + (q . k) u   (own columns; o goes to cb_o at its column position)
//   -> the writer sends the own o tiles to the other CS - 1 cores of the head and collects theirs (cb_osig / cb_orecv)
//   S' = decay S + kcol (x) u  (own columns), overlapping the exchange
//   out = rms_norm(o) (over all 4 o tiles, the same tile order as v1) * w_norm * silu(z)   (own columns)
// With CS = 1 the per-tile operations and their order inside every tile are those of v1, so the results are
// bit-identical; with CS > 1 every tile is still computed by the same instruction sequence from the same operands.

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
                   cb_sel = 11, cb_ypk = 15, cb_out = 16, cb_y = 19, cb_osig = 32, cb_orecv = 33;
// fp32
constexpr uint32_t cb_sdst = 8, cb_smm = 13, cb_qs = 14, cb_state = 17, cb_qk = 18, cb_scal = 20, cb_qn = 21,
                   cb_kn = 22, cb_kc = 23, cb_on = 24, cb_ks = 25, cb_u = 26, cb_o = 28, cb_tmp = 29, cb_decay = 30,
                   cb_beta = 31;

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

// ep2 t3b GDNQ_PRO: scalar_rsqrt with the result into cb o (the rsqrt of every element of the cb_tmp tile)
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

// GDNQ_PRO: out tiles o[0..n) = a[a_off + i] * (column 0 of b[0], broadcast along each row)
void bcast_cols_mul(uint32_t a, uint32_t a_off, uint32_t b, uint32_t o, uint32_t n) {
    CircularBuffer co(o);
    co.reserve_back(n);
    reconfig_data_format(a, b);
    pack_reconfig_data_format(o);
    mul_tiles_init(a, b);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(a, b, a_off + i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    co.push_back(n);
}

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
    constexpr uint32_t QSCALE = get_compile_time_arg_val(0);
    constexpr uint32_t L2_EPS = get_compile_time_arg_val(1);
    constexpr uint32_t ONE = get_compile_time_arg_val(2);
    constexpr uint32_t INV_V = get_compile_time_arg_val(3);
    constexpr uint32_t NORM_EPS = get_compile_time_arg_val(4);
    constexpr uint32_t SP_THRESHOLD = get_compile_time_arg_val(5);
    constexpr uint32_t CS = get_compile_time_arg_val(6);
    constexpr uint32_t NC = 4 / CS;
    const uint32_t ns = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_q, cb_w, cb_ypk);
    CircularBuffer x(cb_x), q(cb_q), w(cb_w), z(cb_z), a(cb_a), b(cb_b), dtb(cb_dtb), aneg(cb_aneg), onw(cb_onw),
        sel(cb_sel), ypk(cb_ypk), out(cb_out), y(cb_y), sdst(cb_sdst), smm(cb_smm), qs(cb_qs), st(cb_state), qk(cb_qk),
        scal(cb_scal), qn(cb_qn), kn(cb_kn), kc(cb_kc), on(cb_on), ks(cb_ks), u(cb_u), o(cb_o), decay(cb_decay),
        beta(cb_beta), osig(cb_osig), orecv(cb_orecv);

#ifdef GDNQ_TS
    cb_wait_front(27, 1);
    const uint32_t tsa = get_tile_address(27, 0);
    // t3b: per slot tk: index 32 * tk + (k - 15) (k = 16 .. 27), index 0 = start
#define CTS(k) PACK((reinterpret_cast<volatile uint32_t*>(tsa)[32 * tk + (k) - 15] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)))
    PACK((reinterpret_cast<volatile uint32_t*>(tsa)[0] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)));
#else
#define CTS(k)
#endif

    constexpr uint32_t NSEL = 8 + NC;
    // ---- gates of both sequences first (inputs from the writer, ready before the conv rows)
    dtb.wait_front(1);
    aneg.wait_front(1);
#ifdef GDNQ_GATES_ALL
    // ep2 t3b: the gates of ALL slots in one pass: the writer put slot r's a / b values in row r of one tile each;
    // decay / beta of every slot -> cb 35 / 36 (row r = slot r), the writer expands row tk into the per-slot uniform
    // tiles of cb_decay / cb_beta (exactly the per-element math of the per-slot pass)
    {
        a.wait_front(1);
        b.wait_front(1);
        CircularBuffer dall(35), ball(36);
        dall.reserve_back(1);
        pack_reconfig_data_format(35);
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
        pack_tile(0, 35);
        tile_regs_release();
        dall.push_back(1);
        ball.reserve_back(1);
        pack_reconfig_data_format(36);
        tile_regs_acquire();
        copy_in(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 36);
        tile_regs_release();
        ball.push_back(1);
        a.pop_front(1);
        b.pop_front(1);
    }
#endif
#ifdef GDNQ_TS
#define PTS(k) PACK((reinterpret_cast<volatile uint32_t*>(tsa)[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)))
#else
#define PTS(k)
#endif
#ifdef GDNQ_PRO
    // ep2 t3b GDNQ_PRO: the state-independent work of all slots up front, rows = slots: the conv of every channel tile
    // (A[k][t] = the k-th conv input rows, W[k][t] = tap k of channel tile t on every row; same ops, same order as the
    // packed per-slot conv), then the q / k l2 norms of all slots (gram diagonal -> rsqrt, the reader moves r,r to r,0,
    // the scale broadcast per row). The reader hands each slot its rows (row 0 tiles) in the chain below.
    constexpr uint32_t NT9 = 8 + NC;
    {
        CircularBuffer pa(37), pw(38), yall(41);
        PTS(300);
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
        PTS(302);
        yall.wait_front(NT9);
        CircularBuffer col(40);
        mm(41, 0, 41, 0, cb_tmp, 4, 1, true);
        tile_rsqrt_to(ONE, L2_EPS, QSCALE, 39);
        col.wait_front(1);
        bcast_cols_mul(41, 0, 40, 42, 4);
        col.pop_front(1);
        PTS(303);
        mm(41, 4, 41, 4, cb_tmp, 4, 1, true);
        tile_rsqrt_to(ONE, L2_EPS, ONE, 39);
        col.wait_front(1);
        bcast_cols_mul(41, 4, 40, 43, 4);
        col.pop_front(1);
        PTS(304);
#ifdef GDNQ_PRO2
        // + k as columns and q . k of all slots (the reader hands out column s / element (s, s))
        CircularBuffer kall(43);
        kall.wait_front(4);
        transpose_block(43, 44, 4);
        CircularBuffer kcall(44);
        kcall.wait_front(4);
        CircularBuffer qall(42);
        qall.wait_front(4);
        mm(42, 0, 44, 0, 45, 4, 1, false);
        PTS(305);
#endif
    }
#endif
    for (uint32_t tk = 0; tk < GDNQ_CHAIN; tk++) {
#ifndef GDNQ_GATES_ALL
    for (uint32_t s = 0; s < ns; s++) {
        // ---- gates
        a.wait_front(1);
        b.wait_front(1);
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
        b.pop_front(1);
    }
#endif
#ifndef GDNQ_PRO
    CTS(27);
    w.wait_front(4);
    {
        // ---- conv on the packed tiles (both sequences)
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
        q.pop_front(3);
        CTS(17);
    }
    sel.wait_front(NSEL * ns);
    ypk.wait_front(1);
    for (uint32_t s = 0; s < ns; s++) {
        // ---- unpack: y_k = E_R @ ypk moves packed row R(t, s) to row 0 (q 0..3, k 4..7, own v -> y 8..8 + NC)
        y.reserve_back(12);  // whole CB blocks (12 tiles): a 9 / 10-tile block would wrap inside the ring
        reconfig_data_format<SrcOrder::Reverse>(cb_sel, cb_ypk);
        pack_reconfig_data_format(cb_y);
        matmul_init(cb_sel, cb_ypk, 0);
        for (uint32_t t = 0; t < 8 + NC; t++) {
            tile_regs_acquire();
            matmul_tiles(cb_sel, cb_ypk, s * NSEL + t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
        }
        y.push_back(12);
        y.wait_front(12);
        CTS(18 + 12 * s);

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
#else
    CTS(27);
    for (uint32_t s = 0; s < ns; s++) {
        y.wait_front(12);
        CTS(18 + 12 * s);
#endif
        qn.wait_front(4);
        kn.wait_front(4);
#ifndef GDNQ_PRO2
        transpose_block(cb_kn, cb_kc, 4);
#endif
        kc.wait_front(4);
        CTS(19 + 12 * s);

        decay.wait_front(1);
        beta.wait_front(1);
        CTS(20 + 12 * s);

        // ---- k S, q S (own columns), q . k
        smm.wait_front(4 * NC);
        mm(cb_kn, 0, cb_smm, 0, cb_ks, 4, NC, false);
        mm(cb_qn, 0, cb_smm, 0, cb_qs, 4, NC, false);
        smm.pop_front(4 * NC);
#ifndef GDNQ_PRO2
        mm(cb_qn, 0, cb_kc, 0, cb_qk, 4, 1, false);
#endif
        ks.wait_front(NC);
        qs.wait_front(NC);
        qk.wait_front(1);
        CTS(21 + 12 * s);

        // ---- u = (v - decay k S) * beta
        u.reserve_back(NC);
        pack_reconfig_data_format(cb_u);
        for (uint32_t j = 0; j < NC; j++) {
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
        u.push_back(NC);
        ks.pop_front(NC);
        u.wait_front(NC);
        CTS(22 + 12 * s);

        // ---- o = decay q S + (q . k) u -> cb_o tiles c0.. (the other columns arrive from the other cores)
#ifdef GDNQ_RMSALL
        osig.reserve_back(1);  // the writer has copied the previous slot's o row
#endif
        o.reserve_back(4);
        pack_reconfig_data_format(cb_o);
        for (uint32_t j = 0; j < NC; j++) {
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
            pack_tile<true>(0, cb_o, c0 + j);
            tile_regs_release();
        }
#ifdef GDNQ_RMSALL
        osig.push_back(1);
#else
        if constexpr (CS > 1) {
            osig.reserve_back(1);
            osig.push_back(1);
        }
#endif
        CTS(23 + 12 * s);

        // ---- S' = decay S + kcol (x) u (own columns) -> cb_state, tiles mi * NC + j
        sdst.wait_front(4 * NC);
        st.reserve_back(4 * NC);
        pack_reconfig_data_format(cb_state);
        for (uint32_t mi = 0; mi < 4; mi++) {
            for (uint32_t j = 0; j < NC; j += 2) {
                const bool two = j + 1 < NC;
                tile_regs_acquire();
                copy_in(cb_sdst, mi * NC + j, 0);
                if (two) {
                    copy_in(cb_sdst, mi * NC + j + 1, 1);
                }
                copy_in(cb_decay, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                if (two) {
                    mul_binary_tile(1, 2, 1);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_kc, cb_u);
                matmul_init(cb_kc, cb_u, 0);
                matmul_tiles(cb_kc, cb_u, mi, j, 0);
                if (two) {
                    matmul_tiles(cb_kc, cb_u, mi, j + 1, 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_state, mi * NC + j);
                if (two) {
                    pack_tile<true>(1, cb_state, mi * NC + j + 1);
                }
                tile_regs_release();
            }
        }
        st.push_back(4 * NC);
        sdst.pop_front(4 * NC);
        CTS(24 + 12 * s);

#ifdef GDNQ_RMSALL
        o.push_back(4);  // (advance the ring: the writer reads the own o tile behind osig)
        o.wait_front(4);
        o.pop_front(4);
#else
        if constexpr (CS > 1) {
            orecv.wait_front(1);  // the other cores' o tiles are in cb_o
            orecv.pop_front(1);
        }
        o.push_back(4);
#endif
        u.pop_front(NC);
        qs.pop_front(NC);
        qk.pop_front(1);
        decay.pop_front(1);
        beta.pop_front(1);
        qn.pop_front(4);
        kn.pop_front(4);
        kc.pop_front(4);

#ifndef GDNQ_RMSALL
        // ---- out = rms_norm(o) * w_norm * silu(z) (own columns)
        o.wait_front(4);
        mm(cb_o, 0, cb_o, 0, cb_tmp, 4, 1, true);
        scalar_rsqrt(INV_V, NORM_EPS, ONE);
        scal.wait_front(1);
        bcast_scalar_mul(cb_o, c0, cb_scal, cb_on, NC);
        scal.pop_front(1);
        o.pop_front(4);
        on.wait_front(NC);
        z.wait_front(NC);
        onw.wait_front(NC);
        CTS(25 + 12 * s);
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
        CTS(26 + 12 * s);
        out.push_back(NC);
        on.pop_front(NC);
        z.pop_front(NC);
#endif
        y.pop_front(12);
    }
#ifndef GDNQ_PRO
    ypk.pop_front(1);
#endif
    }  // slots
#ifdef GDNQ_RMSALL
    // ---- out = rms_norm(o) * w_norm * silu(z) of ALL slots (rows = slots): o_all (cb 46) holds every slot o row
    // of all 4 column tiles (own + the other cores), z / norm weight rows from the reader; the same per-element
    // ops as the per-slot pass (gram diagonal -> rsqrt -> per-row scale by the reader)
    {
        CircularBuffer oall(46), col(40);
        orecv.wait_front(1);
        oall.wait_front(4);
        mm(46, 0, 46, 0, cb_tmp, 4, 1, true);
        tile_rsqrt_to(INV_V, NORM_EPS, ONE, 39);
        col.wait_front(1);
        on.reserve_back(NC);
        reconfig_data_format(46, 40);
        pack_reconfig_data_format(cb_on);
        mul_tiles_init(46, 40);
        for (uint32_t j = 0; j < NC; j++) {
            tile_regs_acquire();
            mul_tiles(46, 40, c0 + j, 0, 0);
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
#endif
#ifndef GDNQ_PRO
    w.pop_front(4);
    sel.pop_front(NSEL * ns);
#endif
    dtb.pop_front(1);
    aneg.pop_front(1);
    onw.pop_front(NC);
}
