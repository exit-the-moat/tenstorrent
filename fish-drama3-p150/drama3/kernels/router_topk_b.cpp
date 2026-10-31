// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: the router top-k of kernels/router_topk.cpp for B decode rows, one core per row b. Input: the router
// scores [1, 1, B, E] bf16 TILE (row b of E / 32 tiles, read as the 64-byte chunks of rows b & ~1, b | 1). Outputs,
// both in L1 (row-sized writes): the top-k scores of row b in row b of one bf16 tile ([1, 1, B, k] TILE; columns
// k..31 = -3e38; core 0 also zeroes rows B..31), and the ids as uint16 at [b * k .. b * k + k) of ONE row-major row
// [1, 1, 1, B * k] (the gather list of sparse_matmul for all rows). Ties keep the lower expert id first.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_scr = 0;

inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr auto in_a = TensorAccessorArgs<3>();
    constexpr auto val_a = TensorAccessorArgs<in_a.next_compile_time_args_offset()>();
    constexpr auto idx_a = TensorAccessorArgs<val_a.next_compile_time_args_offset()>();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t val_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3);
    constexpr uint32_t NT = E / 32;

    const uint32_t tb = get_tile_size(cb_scr);
    const auto in = TensorAccessor(in_a, in_addr, tb);
    const auto val = TensorAccessor(val_a, val_addr, tb);
    const auto idx = TensorAccessor(idx_a, idx_addr, 2 * K * B);

    CircularBuffer scr(cb_scr);
    scr.reserve_back(2);  // tile 0: the output row staging (64 bytes) + ids (2K bytes); tile 1: the row chunks
    const uint32_t out_l1 = scr.get_write_ptr(), s0 = out_l1 + tb;
    const uint32_t xb = rowoff(b & ~1u, 0), sub = (b & 1) * 8;  // in 32-bit words
    for (uint32_t t = 0; t < NT; t++) {
        noc_async_read(in.get_noc_addr(t, xb), s0 + (2 * t) * 64, 64);
        noc_async_read(in.get_noc_addr(t, xb + 512), s0 + (2 * t + 1) * 64, 64);
    }
    noc_async_read_barrier();

    const uint32_t* p = reinterpret_cast<const uint32_t*>(s0);
    uint32_t best_key[K], best_id[K];
    uint16_t best_raw[K];
    for (uint32_t j = 0; j < K; j++) {
        best_key[j] = 0;
        best_id[j] = 0;
        best_raw[j] = 0;
    }
    uint32_t kth = 0;
    for (uint32_t ch = 0; ch < 2 * NT; ch++) {
        const uint32_t* w = p + ch * 16 + sub;  // row b's 32 bytes inside the 64-byte slot
        for (uint32_t i = 0; i < 8; i++) {
            const uint32_t pair = w[i];
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t u = h ? (pair >> 16) : (pair & 0xFFFFu);
                const uint32_t key = ((u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u)) + 1;
                if (key <= kth) {
                    continue;
                }
                const uint32_t e = (ch / 2) * 32 + (ch % 2) * 16 + 2 * i + h;
                uint32_t j = K - 1;
                while (j > 0 && key > best_key[j - 1]) {
                    best_key[j] = best_key[j - 1];
                    best_id[j] = best_id[j - 1];
                    best_raw[j] = best_raw[j - 1];
                    j--;
                }
                best_key[j] = key;
                best_id[j] = e;
                best_raw[j] = static_cast<uint16_t>(u);
                kth = best_key[K - 1];
            }
        }
    }
    // staging: [0, 32) = face 0 row (cols 0..15), [32, 64) = face 1 row (cols 16..31), [64, 64 + 2K) = ids
    volatile tt_l1_ptr uint16_t* o16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1);
    for (uint32_t c = 0; c < 32; c++) {
        o16[c] = c < K ? best_raw[c] : 0xFF62;  // -3e38 past the k scores
    }
    for (uint32_t j = 0; j < K; j++) {
        o16[32 + j] = static_cast<uint16_t>(best_id[j]);
    }
    noc_async_write(out_l1, val.get_noc_addr(0, rowoff(b, 0)), 32);
    noc_async_write(out_l1 + 32, val.get_noc_addr(0, rowoff(b, 1)), 32);
    noc_async_write(out_l1 + 64, idx.get_noc_addr(0, 2 * K * b), 2 * K);
    if (b == 0) {  // the padding rows of the score tile
        for (uint32_t r = B; r < 16; r++) {
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 0)), 32);
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 1)), 32);
        }
        noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, 1024), 512);
        noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, 1536), 512);
    }
    noc_async_write_barrier();
}
