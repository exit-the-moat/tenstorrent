// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3c (QWEN36_GDN_SLOT_K=1): the B=1 GDN prefill scratch of every GDN layer into decode slot `slot` of the batched
// state buffers in ONE program (t2 _write_gdn_slot_from_scratch rebuilt each whole buffer by slice + concat + copy:
// 5 x 4 ops per GDN layer). All tensors DRAM interleaved; one accessor config, runtime addresses and page sizes.
// Runtime args: nseg, then per segment (mode, src, dst, t0, n, param):
//   mode 0: whole fp32 pages (the recurrent state row): dst page param + t <- src page t
//   mode 1: one bf16 row ([1, B, D] conv taps, one tile row): row `param` of dst tile t <- row 0 of src tile t
//           (64-byte row-pair read-modify-write: each tile belongs to one core, the other row of the pair is kept)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto ta = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const uint32_t nseg = get_arg_val<uint32_t>(i++);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t l1 = cb.get_write_ptr();
    for (uint32_t s = 0; s < nseg; s++) {
        const uint32_t mode = get_arg_val<uint32_t>(i++);
        const uint32_t src_addr = get_arg_val<uint32_t>(i++);
        const uint32_t dst_addr = get_arg_val<uint32_t>(i++);
        const uint32_t t0 = get_arg_val<uint32_t>(i++);
        const uint32_t n = get_arg_val<uint32_t>(i++);
        const uint32_t prm = get_arg_val<uint32_t>(i++);
        if (mode == 0) {
            const auto src = TensorAccessor(ta, src_addr, 4096);
            const auto dst = TensorAccessor(ta, dst_addr, 4096);
            for (uint32_t t = t0; t < t0 + n; t++) {
                noc_async_read(src.get_noc_addr(t), l1, 4096);
                noc_async_read_barrier();
                noc_async_write(l1, dst.get_noc_addr(t + prm), 4096);
                noc_async_write_barrier();
            }
        } else {
            const auto src = TensorAccessor(ta, src_addr, 2048);
            const auto dst = TensorAccessor(ta, dst_addr, 2048);
            const uint32_t rp = prm & ~1u;
            const uint32_t po = (rp / 16) * 1024 + (rp % 16) * 32;
            const uint32_t so = (prm & 1u) * 32;
            for (uint32_t t = t0; t < t0 + n; t++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_read(dst.get_noc_addr(t, f * 512 + po), l1 + f * 128, 64);
                    noc_async_read(src.get_noc_addr(t, f * 512), l1 + f * 128 + 64, 64);
                }
                noc_async_read_barrier();
                for (uint32_t f = 0; f < 2; f++) {
                    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + f * 128 + so);
                    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + f * 128 + 64);
                    for (uint32_t w = 0; w < 8; w++) {
                        d[w] = r[w];
                    }
                }
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(l1 + f * 128, dst.get_noc_addr(t, f * 512 + po), 64);
                }
                noc_async_write_barrier();
            }
        }
    }
}
