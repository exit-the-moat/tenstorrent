// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_PF_TCAST=1): bf16 -> fp32 typecast of the GDN out-proj partial, reader: tiles [t0, t1) in batches of G.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto a = TensorAccessorArgs<0>();
    const uint32_t addr = get_arg_val<uint32_t>(0), t0 = get_arg_val<uint32_t>(1), t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t tb = 2048, G = 4;
    const auto x = TensorAccessor(a, addr, tb);
    CircularBuffer cb(0);
    for (uint32_t t = t0; t < t1; t += G) {
        const uint32_t g = (t1 - t) < G ? (t1 - t) : G;
        cb.reserve_back(g);
        const uint32_t p = cb.get_write_ptr();
        for (uint32_t i = 0; i < g; i++) {
            noc_async_read(x.get_noc_addr(t + i, 0), p + i * tb, tb);
        }
        noc_async_read_barrier();
        cb.push_back(g);
    }
}
