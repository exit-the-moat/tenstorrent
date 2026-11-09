// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 pf4 (QWEN36_PF_TCAST=1): bf16 -> fp32 typecast, writer: fp32 tiles [t0, t1).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto a = TensorAccessorArgs<0>();
    const uint32_t addr = get_arg_val<uint32_t>(0), t0 = get_arg_val<uint32_t>(1), t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t tb = 4096, G = 4;
    const auto o = TensorAccessor(a, addr, tb);
    CircularBuffer cb(16);
    for (uint32_t t = t0; t < t1; t += G) {
        const uint32_t g = (t1 - t) < G ? (t1 - t) : G;
        cb.wait_front(g);
        const uint32_t p = cb.get_read_ptr();
        for (uint32_t i = 0; i < g; i++) {
            noc_async_write(p + i * tb, o.get_noc_addr(t + i, 0), tb);
        }
        noc_async_write_barrier();
        cb.pop_front(g);
    }
}
