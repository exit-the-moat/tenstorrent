// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port6: MoE combine for B decode rows, reader. out[b] = sum_j v[b, j] * down[b * K + j][row b] +
// sigmoid(g[b]) * shared[b], over the hidden tiles [t0, t0 + n) of this core. down: the batched sparse_matmul output
// [1, B * K, 32, H] (every slot computed all B rows; only row b of row b's slots counts). Pushes once: the shared-gate
// tile (g_b at (b, 0)) and B * K weight-column tiles (tile s = b * K + j: v[b, j] at (b, 0), zero elsewhere, so a
// column-broadcast multiply keeps row b of slot s only); then per hidden tile c: the B * K down tiles and the
// shared tile c.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_g = 0, cb_w = 1, cb_d = 2, cb_sh = 3, cb_scr = 4;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t S = B * K;
    constexpr auto d_a = TensorAccessorArgs<3>();
    constexpr auto v_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto sh_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<sh_a.next_compile_time_args_offset()>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t sh_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);

    const uint32_t tb = get_tile_size(cb_d);
    const auto d = TensorAccessor(d_a, d_addr, tb);
    const auto v = TensorAccessor(v_a, v_addr, tb);
    const auto sh = TensorAccessor(sh_a, sh_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);

    Noc noc;
    CircularBuffer cg(cb_g), cw(cb_w), cd(cb_d), csh(cb_sh), scr(cb_scr);
    cg.reserve_back(1);
    scr.reserve_back(1);
    cw.reserve_back(S);
    noc.async_read(g, cg, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(v, scr, tb, {.page_id = 0}, {.offset_bytes = 0});
    const uint32_t pw = cw.get_write_ptr();
    for (uint32_t k = 0; k < S * tb; k += 512) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), pw + k, 512);
    }
    noc.async_read_barrier();
    cg.push_back(1);
    // element (r, c) of a bf16 tile in uint16 units
    auto elem = [](uint32_t r, uint32_t c) { return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + (c % 16); };
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pw);
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t j = 0; j < K; j++) {
            dst[(b * K + j) * (tb / 2) + elem(b, 0)] = src[elem(b, j)];
        }
    }
    cw.push_back(S);

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = t0 + i;
        cd.reserve_back(S);
        csh.reserve_back(1);
        for (uint32_t s = 0; s < S; s++) {
            noc.async_read(d, cd, tb, {.page_id = s * HT + c}, {.offset_bytes = s * tb});
        }
        noc.async_read(sh, csh, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cd.push_back(S);
        csh.push_back(1);
    }
}
