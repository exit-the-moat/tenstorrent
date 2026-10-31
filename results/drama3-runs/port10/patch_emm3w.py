src = "kernels/expert_mm3_writer.cpp"
p = "kernels/expert_mm4_writer.cpp"
s = open(src).read()
s = "// drama3 port10: expert_mm4_writer = expert_mm3_writer with the outputs drained while the weights stream (a small\n// output CB; used for B > 16).\n" + s
old_loop_head = """    for (uint32_t b = 0; b < nb + D - 1; b++) {
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, sl = q / NTC, t = q % NTC;
            const uint32_t n = n0 + t, e = uq[s0 + sl * SG];"""
new_loop_head = """    uint32_t wq = 0;  // output tiles written so far (drained while the weights stream: the output CB is small)
    auto put = [&](uint32_t q) {
        const uint32_t u = s0 + (q / NTC) * SG, n = n0 + q % NTC;
        noc_async_write(co.get_read_ptr(), o.get_noc_addr(u * NT + n), tb);
        noc_async_writes_flushed();
        co.pop_front(1);
    };
    for (uint32_t b = 0; b < nb + D - 1; b++) {
        while (wq < nj && co.pages_available_at_front(1)) {
            put(wq++);
        }
        if (b < nb) {
            const uint32_t q = b / NKB, k0 = (b % NKB) * KB, sl = q / NTC, t = q % NTC;
            const uint32_t n = n0 + t, e = uq[s0 + sl * SG];"""
assert old_loop_head in s
s = s.replace(old_loop_head, new_loop_head, 1)
old_tail = """    co.wait_front(nj);  // the output CB holds every job of this core
    for (uint32_t q = 0; q < nj; q++) {
        const uint32_t u = s0 + (q / NTC) * SG, n = n0 + q % NTC;
        noc_async_write(co.get_read_ptr() + q * tb, o.get_noc_addr(u * NT + n), tb);
    }
    noc_async_write_barrier();
    co.pop_front(nj);
"""
new_tail = """    while (wq < nj) {
        co.wait_front(1);
        put(wq++);
    }
    noc_async_write_barrier();
"""
assert old_tail in s
old_res = """            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k + 1) * NT + n), dst + k * tw, tw);"""
new_res = """            while (!cw.pages_reservable_at_back(D * BLK)) {  // never block on the weights with outputs waiting
                if (wq < nj && co.pages_available_at_front(1)) {
                    put(wq++);
                }
            }
            cw.reserve_back(D * BLK);
            noc_async_read_set_trid(1 + b % D);
            const uint32_t dst = w0 + (b % C) * BLK * tw;
            for (uint32_t k = 0; k < BLK; k++) {
                noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k + 1) * NT + n), dst + k * tw, tw);"""
assert old_res in s
s = s.replace(old_res, new_res, 1)
s = s.replace(old_tail, new_tail, 1)
s = s.replace("// K tiles of every job's weights (cb 4), then this core's output tiles:", "// K tiles of every job's weights (cb 4); this core's output tiles are written as the compute packs them:", 1)
open(p, "w").write(s)

p = "tt/moe_fused_decode.py"
s = open(p).read()
old = "            cb(16, max_jobs),\n"
assert old in s
s = s.replace(old, "            cb(16, max_jobs if S <= 128 else 16),  # B > 16: expert_mm4_writer drains it while the weights stream\n", 1)
old = """            kd("expert_mm3_writer.cpp", """
assert old in s
s = s.replace(old, """            kd("expert_mm3_writer.cpp" if S <= 128 else "expert_mm4_writer.cpp", """, 1)
open(p, "w").write(s)
print("ok")
