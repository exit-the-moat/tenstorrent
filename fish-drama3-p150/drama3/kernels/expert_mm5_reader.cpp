// drama3 port10: expert_mm5 = expert_mm2 (ExpertMM v2) with one extra slot NSLOT - 1 that streams the shared
// expert's weights (a second weight tensor, expert 0 of it) instead of a routed expert: the shared expert's
// weight stream runs inside the routed experts' program (DRAMA3_MOE_SHARED_IN_EMM=1). ids holds NSLOT - 1 routed ids.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: ExpertMM v2 (deep weight stream), reader: port5's kernel, but D weight blocks stay in flight per RISC
// (one transaction id per block, a ring of C blocks in the CB) instead of one barrier per block.
// port5:
// out[j, n] = A_j[1, K] @ W[ids[j]][K, n-tile] for the jobs (slot j, output tile n) j0 .. j0 + nj of this core
// (job = j * NT + n). A is either one shared row (gate_up: the token, SHARED_A = 1, read once and kept) or one row per
// slot (down: the SwiGLU output of slot j). The expert ids come from the router top-k (uint16 ROW_MAJOR, in L1).
// Weights stream in blocks of KB K-tiles, split over both data-movement RISCs: this reader takes the even K tiles,
// the writer the odd ones (both NOCs pull weights).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_a = 0, cb_w = 1, cb_ids = 2, cb_stage = 3;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NT = get_compile_time_arg_val(1);  // output N tiles per slot
    constexpr uint32_t KB = get_compile_time_arg_val(2);  // K tiles per weight block
    constexpr uint32_t SHARED_A = get_compile_time_arg_val(3);
    constexpr uint32_t NSLOT = get_compile_time_arg_val(4);
    constexpr uint32_t D = get_compile_time_arg_val(5);  // blocks in flight (<= 15)
    constexpr uint32_t C = get_compile_time_arg_val(6);  // ring slots (blocks) of the weight CB, >= 2 D
    constexpr auto a_a = TensorAccessorArgs<7>();
    constexpr auto w_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto ws_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto i_a = TensorAccessorArgs<ws_a.next_compile_time_args_offset()>();
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t i_addr = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t nj = get_arg_val<uint32_t>(4);
    const uint32_t ws_addr = get_arg_val<uint32_t>(5);

    const uint32_t ta = get_tile_size(cb_a), tw = get_tile_size(cb_w);
    const auto a = TensorAccessor(a_a, a_addr, ta);
    const auto w = TensorAccessor(w_a, w_addr, tw);
    const auto ws = TensorAccessor(ws_a, ws_addr, tw);
    const auto ids = TensorAccessor(i_a, i_addr, 2 * (NSLOT - 1));
    Noc noc;
    CircularBuffer ca(cb_a), cw(cb_w), ci(cb_ids);
    ci.reserve_back(1);
    noc_async_read(ids.get_noc_addr(0, 0), ci.get_write_ptr(), 2 * (NSLOT - 1));
    if constexpr (SHARED_A) {
        // every core needs the whole token row: read only row 0 of its KT tiles (64-byte chunks: rows 0-1 of faces 0
        // and 1) and build the tiles in L1 (MEM_ZEROS fill + 32-byte copies), instead of each core pulling KT full
        // tiles out of the L1 banks (that congested the NOC: more cores were slower)
        CircularBuffer stg(cb_stage);
        ca.reserve_back(KT);
        stg.reserve_back(KT / 16);
        const uint32_t s0 = stg.get_write_ptr(), pa = ca.get_write_ptr();
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t k = 0; k < KT; k++) {
            noc_async_read(a.get_noc_addr(k, 0), s0 + (2 * k) * 64, 64);
            noc_async_read(a.get_noc_addr(k, 512), s0 + (2 * k + 1) * 64, 64);
            const uint32_t p = pa + k * ta;
            noc_async_read(zeros, p + 32, 480);
            noc_async_read(zeros, p + 512 + 32, 480);
            noc_async_read(zeros, p + 1024, 512);
            noc_async_read(zeros, p + 1536, 512);
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < KT; k++) {
            noc_async_read(get_noc_addr(s0 + (2 * k) * 64), pa + k * ta, 32);
            noc_async_read(get_noc_addr(s0 + (2 * k + 1) * 64), pa + k * ta + 512, 32);
        }
    }
    noc.async_read_barrier();
    if constexpr (SHARED_A) {
        ca.push_back(KT);
    }
    volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ci.get_write_ptr());
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nb = nj * NKB;
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, job = j0 + q, j = job / NT, n = job % NT;
            const bool shs = j == NSLOT - 1;  // the shared expert's slot
            const uint32_t e = shs ? 0 : id[j];
            if constexpr (!SHARED_A) {  // A of slot j once per run of jobs of that slot (the compute pops it then)
                if (k0 == 0 && (q == 0 || (job - 1) / NT != j)) {
                    ca.reserve_back(KT);
                    for (uint32_t k = 0; k < KT; k++) {
                        noc.async_read(a, ca, ta, {.page_id = j * KT + k}, {.offset_bytes = k * ta});
                    }
                    noc.async_read_barrier();
                    ca.push_back(KT);
                }
            }
            // ring slot b % C is free once the compute popped block b - C: with b - D + 1 blocks pushed that is
            // exactly "D blocks free"
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {  // the even K tiles (the writer reads the odd ones)
                noc_async_read((shs ? ws : w).get_noc_addr((e * KT + k0 + 2 * k) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
}
