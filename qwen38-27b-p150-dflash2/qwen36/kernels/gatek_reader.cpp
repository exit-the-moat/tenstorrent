// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_GDN_FUSE2=1): GDN prefill output gate, reader. Output tile (r, c) of gated [T, Nv * Dv]: the
// normed head-major o tile (h = c / DT, d = c % DT: index (h * NR + r) * DT + d, i.e. nlp_concat_heads as an index
// map) and the z tile read straight from the fused projection output qkvzab (column tile ZOFF + c, row stride NQ).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_n = 0, cb_z = 1;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);    // output column tiles (Nv * Dv / 32)
    constexpr uint32_t DT = get_compile_time_arg_val(1);    // tiles per head (Dv / 32)
    constexpr uint32_t NQ = get_compile_time_arg_val(2);    // column tiles of qkvzab
    constexpr uint32_t ZOFF = get_compile_time_arg_val(3);  // first z column tile in qkvzab
    constexpr auto n_a = TensorAccessorArgs<4>();
    constexpr auto z_a = TensorAccessorArgs<n_a.next_compile_time_args_offset()>();
    const uint32_t n_addr = get_arg_val<uint32_t>(0);
    const uint32_t z_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    const uint32_t NR = get_arg_val<uint32_t>(4);
    constexpr uint32_t tbn = 4096, tb = 2048;  // n fp32, z bf16
    const auto n = TensorAccessor(n_a, n_addr, tbn);
    const auto z = TensorAccessor(z_a, z_addr, tb);
    CircularBuffer cn(cb_n), cz(cb_z);
    constexpr uint32_t G = 4;  // tiles per read batch (CBs hold 2 G)
    for (uint32_t t = t0; t < t1; t += G) {
        const uint32_t g = (t1 - t) < G ? (t1 - t) : G;
        cn.reserve_back(g);
        cz.reserve_back(g);
        const uint32_t pn = cn.get_write_ptr(), pz = cz.get_write_ptr();
        for (uint32_t i = 0; i < g; i++) {
            const uint32_t r = (t + i) / NC, c = (t + i) % NC;
            const uint32_t h = c / DT, d = c % DT;
            noc_async_read(n.get_noc_addr((h * NR + r) * DT + d, 0), pn + i * tbn, tbn);
            noc_async_read(z.get_noc_addr(r * NQ + ZOFF + c, 0), pz + i * tb, tb);
        }
        noc_async_read_barrier();
        cn.push_back(g);
        cz.push_back(g);
    }
}
