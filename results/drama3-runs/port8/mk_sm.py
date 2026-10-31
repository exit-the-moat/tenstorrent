import os
os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/drama3/kernels")
s = open("router_topk.cpp", encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)


rep("// drama3 port5: one-token MoE router top-k on one core, data movement only",
    "// drama3 port8: router top-k + the softmax over the k scores in one program (DRAMA3_MOE_TOPK_SOFTMAX=1; this\n"
    "// kernel hands the score tile and a max tile to kernels/router_softmax_compute.cpp and writes its result).\n"
    "// port5: one-token MoE router top-k on one core, data movement only")
rep("constexpr uint32_t cb_scr = 0, cb_idx = 1;", "constexpr uint32_t cb_scr = 0, cb_idx = 1, cb_x = 2, cb_m = 3, cb_ones = 4, cb_out = 16;")
rep("    constexpr auto idx_a = TensorAccessorArgs<val_a.next_compile_time_args_offset()>();\n",
    "    constexpr auto idx_a = TensorAccessorArgs<val_a.next_compile_time_args_offset()>();\n"
    "    constexpr auto one_a = TensorAccessorArgs<idx_a.next_compile_time_args_offset()>();\n")
rep("    const uint32_t idx_addr = get_arg_val<uint32_t>(2);\n",
    "    const uint32_t idx_addr = get_arg_val<uint32_t>(2);\n    const uint32_t one_addr = get_arg_val<uint32_t>(3);\n")
rep("    const auto idx = TensorAccessor(idx_a, idx_addr, 2 * K);\n",
    "    const auto idx = TensorAccessor(idx_a, idx_addr, 2 * K);\n    const auto one = TensorAccessor(one_a, one_addr, tb);\n")
rep("    CircularBuffer scr(cb_scr), ci(cb_idx);\n", "    CircularBuffer scr(cb_scr), ci(cb_idx), cx(cb_x), cm(cb_m), co(cb_ones), cout(cb_out);\n")
rep("    ci.reserve_back(1);\n    const uint32_t out_l1 = scr.get_write_ptr(), s0 = out_l1 + tb;\n",
    "    ci.reserve_back(1);\n    cx.reserve_back(1);\n    cm.reserve_back(1);\n    co.reserve_back(1);\n"
    "    const uint32_t out_l1 = cx.get_write_ptr(), s0 = scr.get_write_ptr() + tb, m_l1 = cm.get_write_ptr();\n"
    "    noc.async_read(one, co, tb, {.page_id = 0}, {.offset_bytes = 0});\n")
rep("        noc_async_read(zeros, out_l1 + k, 512);\n", "        noc_async_read(zeros, out_l1 + k, 512);\n        noc_async_read(zeros, m_l1 + k, 512);\n")
rep("        o16[c < 16 ? c : 256 + (c - 16)] = 0xFF62;  // -3e38 past the k scores in row 0\n",
    "        o16[c < 16 ? c : 256 + (c - 16)] = 0xC700;  // -32768 past the k scores in row 0: exp(.. - max) = 0\n")
rep("""    noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(scr), val, tb, {.offset_bytes = 0}, {.page_id = 0});
""", """    volatile tt_l1_ptr uint16_t* m16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_l1);
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
""")
# drop the original idx write + barrier + closing brace that follow
tail = """    noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(ci), idx, 2 * K, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}"""
i = s.rindex(tail)
s = s[:i] + s[i + len(tail):]
s = s.rstrip() + "\n"
open("router_topk_sm.cpp", "w", encoding="utf-8").write(s)

open("router_softmax_compute.cpp", "w").write(r'''// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port8: the softmax over the router's top-k scores (row 0 of X: k scores, then -32768; M: the max in all of
// row 0): E = exp(X - M), S = E @ ones (row 0: the sum in every column), V = E * (1 / S). Rows 1..31 are junk.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 2, cb_m = 3, cb_ones = 4, cb_e = 5, cb_s = 6, cb_out = 16;

void kernel_main() {
    compute_kernel_hw_startup(cb_x, cb_m, cb_out);
    CircularBuffer x(cb_x), m(cb_m), ones(cb_ones), e(cb_e), s(cb_s), out(cb_out);
    x.wait_front(1);
    m.wait_front(1);
    e.reserve_back(1);
    tile_regs_acquire();
    sub_init(cb_x, cb_m);
    sub_tiles(cb_x, cb_m, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_e);
    tile_regs_release();
    e.push_back(1);
    x.pop_front(1);
    m.pop_front(1);
    ones.wait_front(1);
    e.wait_front(1);
    s.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_e, cb_ones, 0);
    matmul_tiles(cb_e, cb_ones, 0, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    s.push_back(1);
    ones.pop_front(1);
    s.wait_front(1);
    out.reserve_back(1);
    tile_regs_acquire();
    mul_init(cb_e, cb_s, false);
    mul_tiles(cb_e, cb_s, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    out.push_back(1);
    e.pop_front(1);
    s.pop_front(1);
}
''')

p = "../tt/moe_fused_decode.py"
t = open(p).read()


def rp(a, b):
    global t
    assert t.count(a) == 1, a
    t = t.replace(a, b)


rp("""        self.core = c

    def __call__(self, scores):""", """        self.core = c
        # port8 DRAMA3_MOE_TOPK_SOFTMAX=1: the softmax over the k scores in the same program (kernels/router_topk_sm.cpp
        # + router_softmax_compute.cpp); __call__ then returns the normalised weights
        self.softmax = os.environ.get("DRAMA3_MOE_TOPK_SOFTMAX", "0") == "1"
        if self.softmax:
            self.ones = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    def __call__(self, scores):""")
rp("""        ct = [self.E, self.k]
        for t in (scores, val, idx):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address()]
""", """        ct = [self.E, self.k]
        for t in (scores, val, idx) + ((self.ones,) if self.softmax else ()):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address()]
        if self.softmax:
            rt[self.core.x][self.core.y] = [scores.buffer_address(), val.buffer_address(), idx.buffer_address(), self.ones.buffer_address()]
""")
rp("""            kernel_source=f"{KDIR}/router_topk.cpp",
""", """            kernel_source=f"{KDIR}/router_topk_sm.cpp" if self.softmax else f"{KDIR}/router_topk.cpp",
""")
rp("""        ttnn.generic_op([scores, val, idx], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
        return val, idx


def install_topk""", """        kernels = [kern]
        ios = [scores, val, idx]
        if self.softmax:
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/router_softmax_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs, compile_time_args=[], runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
            ))
            cbs += [
                ttnn.CBDescriptor(total_size=2048, core_ranges=self.crs,
                                  format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
                for i in (2, 3, 4, 5, 6, 16)
            ]
            ios = [scores, self.ones, val, idx]
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return val, idx


def install_topk""")
rp("""            raw, idx = rk(s)
            s.deallocate(True)
            v = ttnn.softmax(raw, dim=-1)
            raw.deallocate(True)
            return v, idx""", """            raw, idx = rk(s)
            s.deallocate(True)
            if rk.softmax:
                return raw, idx
            v = ttnn.softmax(raw, dim=-1)
            raw.deallocate(True)
            return v, idx""")
open(p, "w").write(t)
print("ok")
