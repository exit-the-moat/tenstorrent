// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10: MoE combine over the DEDUPLICATED expert slots (DRAMA3_MOE_BATCH_DEDUP=1, kernels/expert_mm3_*),
// reader. down [1, NSLOT, 32, H]: slot u = the u-th unique routed expert (first-occurrence order of the B * K ids),
// all B rows. out[b] = sum_j v[b, j] * down[pos[id[b K + j]]][row b] + sigmoid(g[b]) * shared[b]. Pushes once: the
// count U (cb 6), the shared-gate tile and U weight-column tiles (tile u: v[b, j] at (b, 0) for every row b routed
// to expert uq[u]; a row routes to an expert at most once); then per hidden tile c the U down tiles and shared tile c.
// Rows with mask[b] = 0 (idle continuous-batching slots) are left out of the unique list and get the shared expert only.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_g = 0, cb_w = 1, cb_d = 2, cb_sh = 3, cb_scr = 4, cb_cnt = 6, cb_ids = 7;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t S = B * K;
    constexpr auto d_a = TensorAccessorArgs<3>();
    constexpr auto v_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto sh_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<sh_a.next_compile_time_args_offset()>();
    constexpr auto i_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto m_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t sh_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t i_addr = get_arg_val<uint32_t>(6);
    const uint32_t m_addr = get_arg_val<uint32_t>(7);

    const uint32_t tb = get_tile_size(cb_d);
    const auto d = TensorAccessor(d_a, d_addr, tb);
    const auto v = TensorAccessor(v_a, v_addr, tb);
    const auto sh = TensorAccessor(sh_a, sh_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);
    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);
    const auto msk = TensorAccessor(m_a, m_addr, 64);

    Noc noc;
    CircularBuffer cg(cb_g), cw(cb_w), cd(cb_d), csh(cb_sh), scr(cb_scr), cc(cb_cnt), ci(cb_ids);
    ci.reserve_back(1);
    const uint32_t pi = ci.get_write_ptr();
    cg.reserve_back(1);
    scr.reserve_back(1);
    cw.reserve_back(S);
    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);
    noc.async_read(g, cg, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(v, scr, tb, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi);
    volatile tt_l1_ptr uint16_t* pos = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512);
    volatile tt_l1_ptr uint16_t* act = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536);
    for (uint32_t e = 0; e < 256; e++) {
        pos[e] = 0xFFFF;
    }
    uint32_t U = 0;
    for (uint32_t s = 0; s < S; s++) {
        if (act[s / K] != 0 && pos[id[s]] == 0xFFFF) {
            pos[id[s]] = U++;
        }
    }
    const uint32_t pw = cw.get_write_ptr();
    for (uint32_t k = 0; k < U * tb; k += 512) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), pw + k, 512);
    }
    noc.async_read_barrier();
    cc.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr())[0] = U;
    cc.push_back(1);
    cg.push_back(1);
    // element (r, c) of a bf16 tile in uint16 units
    auto elem = [](uint32_t r, uint32_t c) { return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + (c % 16); };
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pw);
    for (uint32_t b = 0; b < B; b++) {
        if (act[b] == 0) {  // an idle slot's row: shared expert only (its output is dropped)
            continue;
        }
        for (uint32_t j = 0; j < K; j++) {
            dst[pos[id[b * K + j]] * (tb / 2) + elem(b, 0)] = src[elem(b, j)];
        }
    }
    cw.push_back(S);  // whole-CB pushes (S tiles, only the first U used): a U-tile block never wraps the ring

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = t0 + i;
        cd.reserve_back(S);
        csh.reserve_back(1);
        for (uint32_t u = 0; u < U; u++) {
            noc.async_read(d, cd, tb, {.page_id = u * HT + c}, {.offset_bytes = u * tb});
        }
        noc.async_read(sh, csh, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cd.push_back(S);
        csh.push_back(1);
    }
}
