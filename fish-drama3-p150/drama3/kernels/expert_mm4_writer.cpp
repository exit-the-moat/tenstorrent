// drama3 port10: expert_mm4_writer = expert_mm3_writer with the outputs drained while the weights stream (a small
// output CB; used for B > 16).
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: ExpertMM v3 writer (kernels/expert_mm3_reader.cpp): the same unique list from the routed ids, the odd
// K tiles of every job's weights (cb 4); this core's output tiles are written as the compute packs them: job (slot u, tile n) -> page u * NT + n of
// the [1, NSLOT, 32, N] result (slots >= U are not written).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_w2 = 4, cb_ids2 = 5, cb_out = 16;

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
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t KB = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t D = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t NTC = get_compile_time_arg_val(6);
    constexpr auto w_a = TensorAccessorArgs<7>();
    constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto o_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();
    constexpr auto m_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr uint32_t KR = get_compile_time_arg_val(m_a.next_compile_time_args_offset());
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t n0 = get_arg_val<uint32_t>(3);
    const uint32_t s0 = get_arg_val<uint32_t>(4);
    const uint32_t SG = get_arg_val<uint32_t>(5);
    const uint32_t m_addr = get_arg_val<uint32_t>(6);
    const uint32_t tw = get_tile_size(cb_w2), tb = get_tile_size(cb_out);
    const auto w = TensorAccessor(w_a, w_addr, tw);
    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    const auto msk = TensorAccessor(m_a, m_addr, 64);
    CircularBuffer cw(cb_w2), ci(cb_ids2), co(cb_out);
    ci.reserve_back(1);
    const uint32_t pi = ci.get_write_ptr();
    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* uq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1024);
    const uint32_t U = dedup(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi), S,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536), KR);
    const uint32_t nsl = U > s0 ? (U - s0 + SG - 1) / SG : 0;
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nj = nsl * NTC, nb = nj * NKB;
    uint32_t wq = 0;  // output tiles written so far (drained while the weights stream: the output CB is small)
    auto put = [&](uint32_t q) {
        const uint32_t u = s0 + (q / NTC) * SG, n = n0 + q % NTC;
        noc_async_write(co.get_read_ptr(), o.get_noc_addr(u * NT + n), tb);
        noc_async_writes_flushed();
        co.pop_front(1);
    };
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        while (wq < nj && co.pages_available_at_front(1)) {
            put(wq++);
        }
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, sl = q / NTC, t = q % NTC;
            const uint32_t n = n0 + t, e = uq[s0 + sl * SG];
            while (!cw.pages_reservable_at_back(D * BLK)) {  // never block on the weights with outputs waiting
                if (wq < nj && co.pages_available_at_front(1)) {
                    put(wq++);
                }
            }
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k + 1) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
    while (wq < nj) {
        co.wait_front(1);
        put(wq++);
    }
    noc_async_write_barrier();
}
