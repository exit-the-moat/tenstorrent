// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: router top-k + the softmax over the k scores in one program (DRAMA3_MOE_TOPK_SOFTMAX=1; this
// kernel hands the score tile and a max tile to kernels/router_softmax_compute.cpp and writes its result).
// port5: one-token MoE router top-k on one core, data movement only (replaces FillPad + TopK + the index
// untilize of the decode router: ~39 us per layer). Input: the router scores [1, 1, 1, E] bf16 TILE (row 0 of
// E / 32 tiles). Outputs: the top-k scores, descending, in row 0 of one bf16 tile ([1, 1, 1, k] TILE; the rest of
// row 0 = -3e38, other rows 0), and their expert ids as uint16 [1, 1, 1, k] ROW_MAJOR (what sparse_matmul reads).
// Ties keep the lower expert id first. bf16 values are compared through an order-preserving 16-bit key.
// Only row 0 of each input tile is read (64-byte chunks: rows 0-1 of faces 0 and 1, into 64-byte aligned slots);
// the output tile's zero rows come from MEM_ZEROS.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_scr = 0, cb_idx = 1, cb_x = 2, cb_m = 3, cb_ones = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr auto in_a = TensorAccessorArgs<2>();
    constexpr auto val_a = TensorAccessorArgs<in_a.next_compile_time_args_offset()>();
    constexpr auto idx_a = TensorAccessorArgs<val_a.next_compile_time_args_offset()>();
    constexpr auto one_a = TensorAccessorArgs<idx_a.next_compile_time_args_offset()>();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t val_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t one_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t NT = E / 32;

    const uint32_t tb = get_tile_size(cb_scr);
    const auto in = TensorAccessor(in_a, in_addr, tb);
    const auto val = TensorAccessor(val_a, val_addr, tb);
    const auto idx = TensorAccessor(idx_a, idx_addr, 2 * K);
    const auto one = TensorAccessor(one_a, one_addr, tb);

    Noc noc;
    CircularBuffer scr(cb_scr), ci(cb_idx), cx(cb_x), cm(cb_m), co(cb_ones), cout(cb_out);
    scr.reserve_back(2);  // tile 0: the output tile; tile 1: the row-0 chunks (2 * NT slots of 64 bytes)
    ci.reserve_back(1);
    cx.reserve_back(1);
    cm.reserve_back(1);
    co.reserve_back(1);
    const uint32_t out_l1 = cx.get_write_ptr(), s0 = scr.get_write_ptr() + tb, m_l1 = cm.get_write_ptr();
    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});
    for (uint32_t t = 0; t < NT; t++) {
        noc_async_read(in.get_noc_addr(t, 0), s0 + (2 * t) * 64, 64);
        noc_async_read(in.get_noc_addr(t, 512), s0 + (2 * t + 1) * 64, 64);
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t k = 0; k < tb; k += 512) {
        noc_async_read(zeros, out_l1 + k, 512);
        noc_async_read(zeros, m_l1 + k, 512);
    }
    noc.async_read_barrier();

    // chunk (t, f) holds columns 16 f .. 16 f + 15 of tile t, i.e. experts 32 t + 16 f ..; two bf16 per word
    const uint32_t* p = reinterpret_cast<const uint32_t*>(s0);
    uint32_t best_key[K], best_id[K];
    uint16_t best_raw[K];
    for (uint32_t j = 0; j < K; j++) {
        best_key[j] = 0;
        best_id[j] = 0;
        best_raw[j] = 0;
    }
    uint32_t kth = 0;  // best_key[K - 1]
    for (uint32_t ch = 0; ch < 2 * NT; ch++) {
        const uint32_t* w = p + ch * 16;  // 64-byte slot, the first 32 bytes are row 0
        for (uint32_t i = 0; i < 8; i++) {
            const uint32_t pair = w[i];
            for (uint32_t h = 0; h < 2; h++) {
                const uint32_t u = h ? (pair >> 16) : (pair & 0xFFFFu);
                // order-preserving key: positive -> set the sign bit, negative -> invert all bits; +1 so 0 = "empty"
                const uint32_t key = ((u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u)) + 1;
                if (key <= kth) {
                    continue;  // not strictly better than the k-th: a tie keeps the lower id already in the list
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
    volatile tt_l1_ptr uint16_t* o16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1);
    volatile tt_l1_ptr uint16_t* iw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ci.get_write_ptr());
    for (uint32_t c = K; c < 32; c++) {
        o16[c < 16 ? c : 256 + (c - 16)] = 0xC700;  // -32768 past the k scores in row 0: exp(.. - max) = 0
    }
    for (uint32_t j = 0; j < K; j++) {
        o16[j] = best_raw[j];  // K <= 16: face 0
        iw[j] = static_cast<uint16_t>(best_id[j]);
    }
    volatile tt_l1_ptr uint16_t* m16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_l1);
    for (uint32_t c = 0; c < 32; c++) {
        m16[c < 16 ? c : 256 + (c - 16)] = best_raw[0];  // the max score in all of row 0
    }
    cx.push_back(1);
    cm.push_back(1);
    co.push_back(1);
    noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(ci), idx, 2 * K, {.offset_bytes = 0}, {.page_id = 0});
    cout.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cout), val, tb, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    cout.pop_front(1);
}
