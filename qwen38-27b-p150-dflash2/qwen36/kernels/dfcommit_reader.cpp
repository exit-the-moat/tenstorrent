// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3 (Exit the Moat, Episode 2): the DFlash2 commit of one GDN layer as ONE data-movement program
// (DF_COMMIT_K=1): base <- bufs[a] (the fp32 state after the accepted slot a, tile for tile) and the conv history
// hist [1, 3, D] <- rows a + 1 .. a + 3 of the persistent verify window X [1, 3 + 2T, D] (one tile row: whole X tiles
// are read, the 3 rows moved inside L1, whole hist tiles written; the hist padding rows are zeros). This core:
// state tiles [s0, s0 + sn), channel tiles [h0, h0 + hn). Pure copies: bit-identical to the ttnn copy / slice ops.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t A = GDNC_A;  // the accepted slot
    constexpr auto bufa = TensorAccessorArgs<0>();
    constexpr auto basea = TensorAccessorArgs<bufa.next_compile_time_args_offset()>();
    constexpr auto xpa = TensorAccessorArgs<basea.next_compile_time_args_offset()>();
    constexpr auto hista = TensorAccessorArgs<xpa.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t buf_addr = get_arg_val<uint32_t>(i++);
    const uint32_t base_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t hist_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s0 = get_arg_val<uint32_t>(i++);
    const uint32_t sn = get_arg_val<uint32_t>(i++);
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t hn = get_arg_val<uint32_t>(i++);

    constexpr uint32_t tf = 4096, tb = 2048, NB = 4;
    const auto buf = TensorAccessor(bufa, buf_addr, tf);
    const auto base = TensorAccessor(basea, base_addr, tf);
    const auto xp = TensorAccessor(xpa, xp_addr, tb);
    const auto hist = TensorAccessor(hista, hist_addr, tb);
    CircularBuffer cs(0), ch(1);
    cs.reserve_back(NB);
    const uint32_t ps = cs.get_write_ptr();
    ch.reserve_back(2);
    const uint32_t px = ch.get_write_ptr(), ph = px + tb;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    // ---- state tiles, NB in flight
    for (uint32_t k = 0; k < sn; k += NB) {
        const uint32_t n = (sn - k) < NB ? (sn - k) : NB;
        for (uint32_t j = 0; j < n; j++) {
            noc_async_read(buf.get_noc_addr(s0 + k + j), ps + j * tf, tf);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < n; j++) {
            noc_async_write(ps + j * tf, base.get_noc_addr(s0 + k + j), tf);
        }
        noc_async_write_barrier();
    }
    // ---- history tiles
    for (uint32_t c = h0; c < h0 + hn; c++) {
        noc_async_read(xp.get_noc_addr(c), px, tb);
        for (uint32_t q4 = 0; q4 < 4; q4++) {
            noc_async_read(zeros, ph + q4 * 512, 512);
        }
        noc_async_read_barrier();
        for (uint32_t f = 0; f < 2; f++) {
            for (uint32_t r = 0; r < 3; r++) {
                noc_async_read(get_noc_addr(px + f * 512 + (A + 1 + r) * 32), ph + f * 512 + r * 32, 32);
            }
        }
        noc_async_read_barrier();
        noc_async_write(ph, hist.get_noc_addr(c), tb);
        noc_async_write_barrier();
    }
}
