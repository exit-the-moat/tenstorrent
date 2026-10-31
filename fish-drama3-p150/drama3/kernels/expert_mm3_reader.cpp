// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: ExpertMM v3 (DRAMA3_MOE_BATCH_DEDUP=1), reader: the expert matmuls of B decode rows over the
// DEDUPLICATED routed experts. ids = the B * K routed ids (router_topk_b's uint16 row); every RISC builds the same
// unique list (first-occurrence order) and its count U at run time, so a trace stays valid for any routing.
// out[u][32 rows, n] = A_u @ W[uq[u]][K, n-tile], A_u = the shared B-row token tile row (gate_up, SHARED_A = 1,
// rows b = decode rows) or slot u's own rows (down: the SwiGLU output [1, NSLOT, 32, K]).
// Core (n0, s0): output tiles n0 .. n0 + NTC - 1 of the slots u = s0, s0 + SG, .. < U (slot-major jobs; A of a
// non-shared slot is read once per slot run). Weights stream as in v2 (port8): D blocks of KB K tiles in flight per
// RISC, a ring of C blocks, this reader the even K tiles, the writer the odd ones. A's reads use transaction id 15,
// so waiting for A never drains the weight stream.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_a = 0, cb_w = 1, cb_ids = 2, cb_cnt = 6;
constexpr uint32_t TRID_A = 15;

// the unique experts of id[0..S) in first-occurrence order -> uq[0..U), pos[e] = u (E <= 256)
inline uint32_t dedup(volatile tt_l1_ptr uint16_t* id, uint32_t S, volatile tt_l1_ptr uint16_t* pos,
                      volatile tt_l1_ptr uint16_t* uq, volatile tt_l1_ptr uint16_t* act, uint32_t K) {
    for (uint32_t e = 0; e < 256; e++) {
        pos[e] = 0xFFFF;
    }
    uint32_t U = 0;
    for (uint32_t s = 0; s < S; s++) {
        if (act[s / K] == 0) {  // an idle slot's row: its experts are not computed
            continue;
        }
        const uint32_t e = id[s];
        if (pos[e] == 0xFFFF) {
            pos[e] = U;
            uq[U++] = e;
        }
    }
    return U;
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);   // K tiles
    constexpr uint32_t NT = get_compile_time_arg_val(1);   // output N tiles per slot (whole tensor)
    constexpr uint32_t KB = get_compile_time_arg_val(2);   // K tiles per weight block
    constexpr uint32_t SHARED_A = get_compile_time_arg_val(3);
    constexpr uint32_t S = get_compile_time_arg_val(4);    // routed ids (B * K)
    constexpr uint32_t D = get_compile_time_arg_val(5);    // blocks in flight (<= 14)
    constexpr uint32_t C = get_compile_time_arg_val(6);    // ring slots (blocks), >= 2 D
    constexpr uint32_t NTC = get_compile_time_arg_val(7);  // output tiles per slot on this core
    constexpr auto a_a = TensorAccessorArgs<8>();
    constexpr auto w_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto m_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();
    constexpr uint32_t KR = get_compile_time_arg_val(m_a.next_compile_time_args_offset());  // routed per row
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t i_addr = get_arg_val<uint32_t>(2);
    const uint32_t n0 = get_arg_val<uint32_t>(3);
    const uint32_t s0 = get_arg_val<uint32_t>(4);
    const uint32_t SG = get_arg_val<uint32_t>(5);
    const uint32_t m_addr = get_arg_val<uint32_t>(6);

    const uint32_t ta = get_tile_size(cb_a), tw = get_tile_size(cb_w);
    const auto a = TensorAccessor(a_a, a_addr, ta);
    const auto w = TensorAccessor(w_a, w_addr, tw);
    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);
    const auto msk = TensorAccessor(m_a, m_addr, 64);
    CircularBuffer ca(cb_a), cw(cb_w), ci(cb_ids), cc(cb_cnt);
    ci.reserve_back(1);
    const uint32_t pi = ci.get_write_ptr();
    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);  // the active-row mask (32 x uint16)
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* uq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1024);
    const uint32_t U = dedup(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi), S,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536), KR);
    cc.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr())[0] = U;
    cc.push_back(1);
    const uint32_t nsl = U > s0 ? (U - s0 + SG - 1) / SG : 0;

    if constexpr (SHARED_A) {
        ca.reserve_back(KT);
        for (uint32_t k = 0; k < KT; k++) {
            noc_async_read(a.get_noc_addr(k), ca.get_write_ptr() + k * ta, ta);
        }
        noc_async_read_barrier();
        ca.push_back(KT);
    }
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nb = nsl * NTC * NKB;
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, sl = q / NTC, t = q % NTC;
            const uint32_t u = s0 + sl * SG, n = n0 + t, e = uq[u];
            if constexpr (!SHARED_A) {
                if (t == 0 && k0 == 0) {  // slot u's rows, once per slot run (the compute pops them after NTC jobs)
                    ca.reserve_back(KT);
                    noc_async_read_set_trid(TRID_A);
                    const uint32_t pa = ca.get_write_ptr();
                    for (uint32_t k = 0; k < KT; k++) {
                        noc_async_read(a.get_noc_addr(u * KT + k), pa + k * ta, ta);
                    }
                    noc_async_read_barrier_with_trid(TRID_A);
                    ca.push_back(KT);
                }
            }
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
}
