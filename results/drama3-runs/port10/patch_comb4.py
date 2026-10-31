KD = "kernels/"
r = open(KD + "moe_combine_b3_reader.cpp").read()
old = """    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = t0 + i;
        cd.reserve_back(S);
        csh.reserve_back(1);
        for (uint32_t u = 0; u < U; u++) {
            noc.async_read(d, cd, tb, {.page_id = u * HT + c}, {.offset_bytes = u * tb});
        }
        noc.async_read(sh, csh, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cd.push_back(S);
        csh.push_back(1);
    }"""
new = """    for (uint32_t i = 0; i < n; i++) {
        const uint32_t c = t0 + i;
        csh.reserve_back(1);
        noc.async_read(sh, csh, tb, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        csh.push_back(1);
        for (uint32_t u0 = 0; u0 < U; u0 += CH) {  // the U down tiles in chunks of CH (the last one partly unused)
            cd.reserve_back(CH);
            for (uint32_t u = u0; u < U && u < u0 + CH; u++) {
                noc.async_read(d, cd, tb, {.page_id = u * HT + c}, {.offset_bytes = (u - u0) * tb});
            }
            noc.async_read_barrier();
            cd.push_back(CH);
        }
    }"""
assert old in r
r = r.replace(old, new, 1)
r = r.replace("constexpr uint32_t cb_g = 0,", "constexpr uint32_t CH = 16;  // down tiles per push (b4: B > 16)\nconstexpr uint32_t cb_g = 0,", 1)
r = "// drama3 port10: moe_combine_b4 = b3 with the down tiles pushed in chunks of 16 (L1 for B > 16).\n" + r
open(KD + "moe_combine_b4_reader.cpp", "w").write(r)

c = open(KD + "moe_combine_b3_compute.cpp").read()
old = """    for (uint32_t i = 0; i < n; i++) {
        d.wait_front(S);
        sh.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        mul_bcast_cols_init(cb_sh, cb_gs);
        mul_tiles_bcast_cols(cb_sh, cb_gs, 0, 0, 0);
        for (uint32_t u = 0; u < U; u++) {
            mul_bcast_cols_init(cb_d, cb_w);
            mul_tiles_bcast_cols(cb_d, cb_w, u, u, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        d.pop_front(S);
        sh.pop_front(1);
    }"""
new = """    for (uint32_t i = 0; i < n; i++) {
        sh.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        mul_bcast_cols_init(cb_sh, cb_gs);
        mul_tiles_bcast_cols(cb_sh, cb_gs, 0, 0, 0);
        for (uint32_t u0 = 0; u0 < U; u0 += CH) {
            d.wait_front(CH);
            for (uint32_t u = u0; u < U && u < u0 + CH; u++) {
                mul_bcast_cols_init(cb_d, cb_w);
                mul_tiles_bcast_cols(cb_d, cb_w, u - u0, u, 0);
            }
            d.pop_front(CH);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        sh.pop_front(1);
    }"""
assert old in c
c = c.replace(old, new, 1)
c = c.replace("constexpr uint32_t cb_g = 0,", "constexpr uint32_t CH = 16;\nconstexpr uint32_t cb_g = 0,", 1)
c = "// drama3 port10: moe_combine_b4 compute = b3 with the down tiles consumed in chunks of 16.\n" + c
open(KD + "moe_combine_b4_compute.cpp", "w").write(c)

p = "tt/moe_fused_decode.py"
s = open(p).read()
old = """            kd("moe_combine_b3_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("moe_combine_b3_compute.cpp", [S, self.per], [],"""
new = """            kd(f"moe_combine_b{v}_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd(f"moe_combine_b{v}_compute.cpp", [S, self.per], [],"""
assert old in s
s = s.replace(old, new, 1)
old = """        sizes = {0: (1, 2048), 1: (S, 2048), 2: (S, 2048), 3: (1, 2048), 4: (1, 2048), 5: (1, 2048), 6: (1, 64), 7: (1, 2048), 16: (2, 2048)}"""
new = """        v = 3 if S <= 128 else 4  # B > 16: the down tiles in chunks of 16 (b4)
        sizes = {0: (1, 2048), 1: (S, 2048), 2: (S if v == 3 else 32, 2048), 3: (1, 2048), 4: (1, 2048), 5: (1, 2048), 6: (1, 64), 7: (1, 2048), 16: (2, 2048)}"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("ok")
