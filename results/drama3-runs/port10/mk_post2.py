c = open("attn_post_compute.cpp").read()
c = c.replace("// drama3 port5: fused one-token", "// drama3 port10: attn_post_compute.cpp looped over the n rows of this core (runtime arg 0; attn_post_b2_*).\n// drama3 port5: fused one-token", 1)
c = c.replace("    CircularBuffer ca(cb_a), cg(cb_g), cs(cb_s), co(cb_out);\n", "    CircularBuffer ca(cb_a), cg(cb_g), cs(cb_s), co(cb_out);\n    const uint32_t n = get_arg_val<uint32_t>(0);\n    for (uint32_t it = 0; it < n; it++) {\n", 1)
c = c.rstrip()
assert c.endswith("}")
c = c[:-1] + "    }\n}\n"
open("attn_post_b2_compute.cpp", "w").write(c)
inc = '#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/circular_buffer.h"\n#include "api/tensor/noc_traits.h"\n'
hdr = "// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n// SPDX-License-Identifier: Apache-2.0\n//\n"
open("attn_post_b2_reader.cpp", "w").write(hdr + """// drama3 port10: attn_post_b_reader.cpp with several rows per core (B > cores / 8): core (j, g0) reads tiles 8 b + j
// of the SDPA output and the packed gate for b = g0, g0 + G, .. < B.

""" + inc + """
constexpr uint32_t cb_a = 0, cb_g = 1;

void kernel_main() {
    constexpr auto a_a = TensorAccessorArgs<0>();
    constexpr auto g_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t g0 = get_arg_val<uint32_t>(3);
    const uint32_t G = get_arg_val<uint32_t>(4);
    const uint32_t B = get_arg_val<uint32_t>(5);
    const uint32_t tb = get_tile_size(cb_a);
    const auto a = TensorAccessor(a_a, a_addr, tb);
    const auto g = TensorAccessor(g_a, g_addr, tb);
    Noc noc;
    CircularBuffer ca(cb_a), cg(cb_g);
    for (uint32_t b = g0; b < B; b += G) {
        ca.reserve_back(1);
        cg.reserve_back(1);
        noc.async_read(a, ca, tb, {.page_id = 8 * b + j}, {.offset_bytes = 0});
        noc.async_read(g, cg, tb, {.page_id = 8 * b + j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        ca.push_back(1);
        cg.push_back(1);
    }
}
""")
open("attn_post_b2_writer.cpp", "w").write(hdr + """// drama3 port10: attn_post_b_writer.cpp with several rows per core (B > cores / 8; B <= 32): rows b = g0, g0 + G, ..
// < B. The g0 = 0 cores zero the padding rows B..31 row by row: rows >= 16 may be other cores' sequences, so faces
// 2 / 3 are not zeroed as a whole.

""" + inc + """
constexpr uint32_t cb_out = 16;

inline uint32_t rowoff(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t g0 = get_arg_val<uint32_t>(2);
    const uint32_t G = get_arg_val<uint32_t>(3);
    const uint32_t B = get_arg_val<uint32_t>(4);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o = TensorAccessor(o_a, o_addr, tb);
    CircularBuffer co(cb_out);
    if (g0 == 0) {
        for (uint32_t h = 0; h < NH; h++) {
            for (uint32_t r = B; r < 32; r++) {
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 0)), 32);
                noc_async_write(MEM_ZEROS_BASE, o.get_noc_addr(8 * h + j, rowoff(r, 1)), 32);
            }
        }
    }
    for (uint32_t b = g0; b < B; b += G) {
        co.wait_front(1);
        const uint32_t src = co.get_read_ptr();
        for (uint32_t h = 0; h < NH; h++) {
            noc_async_write(src + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 0)), 32);
            noc_async_write(src + 512 + h * 32, o.get_noc_addr(8 * h + j, rowoff(b, 1)), 32);
        }
        noc_async_write_barrier();
        co.pop_front(1);
    }
    noc_async_write_barrier();
}
""")
