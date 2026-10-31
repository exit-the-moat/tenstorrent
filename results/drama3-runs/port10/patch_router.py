k = open("kernels/router_topk_b.cpp").read()
old = """        for (uint32_t r = B; r < 16; r++) {
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 0)), 32);
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 1)), 32);
        }
        noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, 1024), 512);
        noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, 1536), 512);
"""
new = """        for (uint32_t r = B; r < 32; r++) {  // row by row: rows >= 16 may belong to other cores
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 0)), 32);
            noc_async_write(MEM_ZEROS_BASE, val.get_noc_addr(0, rowoff(r, 1)), 32);
        }
"""
assert old in k
k = "// drama3 port10: router_topk_b2 = router_topk_b for 16 < B <= 32 (padding rows zeroed row by row).\n" + k.replace(old, new, 1)
open("kernels/router_topk_b2.cpp", "w").write(k)
p = "tt/moe_fused_decode.py"
s = open(p).read()
old = """            kernel_source=f"{KDIR}/router_topk_b.cpp", source_type"""
new = """            kernel_source=f"{KDIR}/router_topk_b{'2' if B > 16 else ''}.cpp", source_type"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("ok")
