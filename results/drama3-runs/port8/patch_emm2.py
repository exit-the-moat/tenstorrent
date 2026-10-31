p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/moe_fused_decode.py"
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)


rep("""        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32
        KB = min(self.kb, KT)
        assert KT % KB == 0
""", """        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32
        KB = min(self.kb, KT)
        assert KT % KB == 0
        # port8 DRAMA3_MOE_EXPERT_MM_V=2: D weight blocks in flight per RISC, a ring of C = 2 D blocks, outputs at the end
        v2 = os.environ.get("DRAMA3_MOE_EXPERT_MM_V", "1") == "2"
        D = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_D", "4"))
        C = 2 * D
""")
rep("""        kernels = [
            kd("expert_mm_reader.cpp", [KT, NT, KB, int(shared_a), nslot] + acc(a, w, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd("expert_mm_writer.cpp", [KT, NT, KB, nslot] + acc(w, ids, out), rw, ttnn.WriterConfigDescriptor()),
""", """        dc = [D, C] if v2 else []
        sfx = "2" if v2 else ""
        kernels = [
            kd(f"expert_mm{sfx}_reader.cpp", [KT, NT, KB, int(shared_a), nslot] + dc + acc(a, w, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd(f"expert_mm{sfx}_writer.cpp", [KT, NT, KB, nslot] + dc + acc(w, ids, out), rw, ttnn.WriterConfigDescriptor()),
""")
rep("""            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * KB, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * KB, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=4, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
""", """            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * (C * KB // 2 if v2 else KB), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
            ttnn.CBDescriptor(total_size=TILE_BYTES[wt] * (C * KB // 2 if v2 else KB), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=4, data_format=w.dtype, page_size=TILE_BYTES[wt])]),
""")
rep("""            ttnn.CBDescriptor(total_size=2048 * 2, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
""", """            ttnn.CBDescriptor(total_size=2048 * (max(2, per + 1) if v2 else 2), core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
""")
open(p, "w").write(s)
print("patched")
