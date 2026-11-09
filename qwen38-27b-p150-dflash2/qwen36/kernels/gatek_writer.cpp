// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_GDN_FUSE2=1): GDN prefill output gate, writer: tile t of gated [T, Nv * Dv] (fp32, row-major tile order).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto o_a = TensorAccessorArgs<0>();
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t tb = 4096;  // fp32 output tiles
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out);
    constexpr uint32_t G = 4;
    for (uint32_t t = t0; t < t1; t += G) {
        const uint32_t g = (t1 - t) < G ? (t1 - t) : G;
        co.wait_front(g);
        const uint32_t p = co.get_read_ptr();
        for (uint32_t i = 0; i < g; i++) {
            noc_async_write(p + i * tb, o.get_noc_addr(t + i, 0), tb);
        }
        noc_async_write_barrier();
        co.pop_front(g);
    }
}
