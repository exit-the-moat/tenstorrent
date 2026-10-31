import os
os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/drama3/kernels")
r = open("expert_mm_reader.cpp").read()
i = r.index("    volatile tt_l1_ptr uint16_t* id")
head = r[:i]
head = head.replace(
    "drama3 port5: one-token indexed expert matmul (the decode MoE's sparse_matmul with k routed experts), reader.",
    "drama3 port8: ExpertMM v2 (deep weight stream), reader: port5's kernel, but D weight blocks stay in flight per RISC\n"
    "// (one transaction id per block, a ring of C blocks in the CB) instead of one barrier per block.\n// port5:")
head = head.replace("    constexpr uint32_t NSLOT = get_compile_time_arg_val(4);\n",
                    "    constexpr uint32_t NSLOT = get_compile_time_arg_val(4);\n"
                    "    constexpr uint32_t D = get_compile_time_arg_val(5);  // blocks in flight (<= 15)\n"
                    "    constexpr uint32_t C = get_compile_time_arg_val(6);  // ring slots (blocks) of the weight CB, >= 2 D\n")
head = head.replace("TensorAccessorArgs<5>()", "TensorAccessorArgs<7>()")
body = r'''    volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ci.get_write_ptr());
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nb = nj * NKB;
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, job = j0 + q, j = job / NT, n = job % NT;
            const uint32_t e = id[j];
            if constexpr (!SHARED_A) {
                if (k0 == 0) {
                    ca.reserve_back(KT);
                    for (uint32_t k = 0; k < KT; k++) {
                        noc.async_read(a, ca, ta, {.page_id = j * KT + k}, {.offset_bytes = k * ta});
                    }
                    noc.async_read_barrier();
                    ca.push_back(KT);
                }
            }
            // ring slot b % C is free once the compute popped block b - C: with b - D + 1 blocks pushed that is
            // exactly "D blocks free"
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {  // the even K tiles (the writer reads the odd ones)
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
}
'''
open("expert_mm2_reader.cpp", "w").write(head + body)

w = open("expert_mm_writer.cpp", encoding="latin-1").read()
i = w.index("    volatile tt_l1_ptr uint16_t* id")
head = w[:i].replace("drama3 port5: one-token indexed expert matmul, writer:",
                     "drama3 port8: ExpertMM v2 writer (deep weight stream as expert_mm2_reader.cpp; the outputs are written at\n"
                     "// the end, so the stream never waits on the compute). port5:")
head = head.replace("    constexpr uint32_t NSLOT = get_compile_time_arg_val(3);\n",
                    "    constexpr uint32_t NSLOT = get_compile_time_arg_val(3);\n"
                    "    constexpr uint32_t D = get_compile_time_arg_val(4);\n"
                    "    constexpr uint32_t C = get_compile_time_arg_val(5);\n")
head = head.replace("TensorAccessorArgs<4>()", "TensorAccessorArgs<6>()")
body = r'''    volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ci.get_write_ptr());
    constexpr uint32_t BLK = KB / 2, NKB = KT / KB;
    const uint32_t w0 = cw.get_write_ptr(), nb = nj * NKB;
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, job = j0 + q, j = job / NT, n = job % NT;
            const uint32_t e = id[j];
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k + 1) * NT + n), dst + k * tw, tw);
            }
        }
        if (b >= D - 1) {
            noc_async_read_barrier_with_trid(1 + (b - (D - 1)) % D);
            cw.push_back(BLK);
        }
    }
    noc_async_read_set_trid(0);
    co.wait_front(nj);  // the output CB holds every job of this core
    for (uint32_t q = 0; q < nj; q++) {
        noc_async_write(co.get_read_ptr() + q * tb, o.get_noc_addr(j0 + q), tb);
    }
    noc_async_write_barrier();
    co.pop_front(nj);
}
'''
open("expert_mm2_writer.cpp", "w", encoding="latin-1").write(head + body)
print("ok")
