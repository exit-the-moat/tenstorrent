import os
os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/drama3/kernels")
s = open("expert_mm2_reader.cpp").read()
a = """            if constexpr (!SHARED_A) {
                if (k0 == 0) {
                    ca.reserve_back(KT);"""
b = """            if constexpr (!SHARED_A) {  // A of slot j once per run of jobs of that slot (the compute pops it then)
                if (k0 == 0 && (q == 0 || (job - 1) / NT != j)) {
                    ca.reserve_back(KT);"""
assert s.count(a) == 1
s = s.replace(a, b)
open("expert_mm2_reader.cpp", "w").write(s)

c = open("expert_mm_compute.cpp").read()
c = c.replace("// drama3 port5: one-token indexed expert matmul, compute:",
              "// drama3 port8: ExpertMM v2 compute: as port5's, but A (non-shared) is held per run of jobs of one slot.\n// port5:")
c = c.replace("    const uint32_t nj = get_arg_val<uint32_t>(0);\n",
              "    const uint32_t nj = get_arg_val<uint32_t>(0);\n    const uint32_t j0 = get_arg_val<uint32_t>(1);\n"
              "    constexpr uint32_t NT = get_compile_time_arg_val(3);\n")
a = """        if constexpr (!SHARED_A) {
            ca.wait_front(KT);
        }
        tile_regs_acquire();"""
b = """        const uint32_t job = j0 + q;
        const bool new_slot = q == 0 || (job - 1) / NT != job / NT;
        if constexpr (!SHARED_A) {
            if (new_slot) {
                if (q > 0) {
                    ca.pop_front(KT);
                }
                ca.wait_front(KT);
            }
        }
        tile_regs_acquire();"""
assert c.count(a) == 1
c = c.replace(a, b)
a = """        co.push_back(1);
        if constexpr (!SHARED_A) {
            ca.pop_front(KT);
        }
    }
    if constexpr (SHARED_A) {
        ca.pop_front(KT);
    }"""
b = """        co.push_back(1);
    }
    ca.pop_front(KT);"""
assert c.count(a) == 1
c = c.replace(a, b)
open("expert_mm2_compute.cpp", "w").write(c)

p = "../tt/moe_fused_decode.py"
t = open(p).read()
a = """                kernel_source=f"{KDIR}/expert_mm_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs, compile_time_args=[KT, KB, int(shared_a)], runtime_args=rc,"""
b = """                kernel_source=f"{KDIR}/expert_mm{sfx}_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.crs, compile_time_args=[KT, KB, int(shared_a)] + ([NT] if v2 else []), runtime_args=rc,"""
assert t.count(a) == 1
t = t.replace(a, b)
a = """            rc[c.x][c.y] = [nj]
"""
b = """            rc[c.x][c.y] = [nj, j0] if v2 else [nj]
"""
assert t.count(a) == 1
t = t.replace(a, b)
a = """            ttnn.CBDescriptor(total_size=2048 * (KT if shared_a else 2 * KT), core_ranges=self.crs,"""
b = """            ttnn.CBDescriptor(total_size=2048 * (KT if shared_a else (3 if v2 else 2) * KT), core_ranges=self.crs,"""
assert t.count(a) == 1
t = t.replace(a, b)
open(p, "w").write(t)
print("ok")
