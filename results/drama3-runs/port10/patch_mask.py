import re

KD = "kernels/"
# ---- expert_mm3 reader / writer: dedup over the active rows only (mask: uint16 [B] row-major, 1 = active)
for f in ("expert_mm3_reader.cpp", "expert_mm3_writer.cpp"):
    s = open(KD + f).read()
    s = s.replace("""inline uint32_t dedup(volatile tt_l1_ptr uint16_t* id, uint32_t S, volatile tt_l1_ptr uint16_t* pos,
                      volatile tt_l1_ptr uint16_t* uq) {""", """inline uint32_t dedup(volatile tt_l1_ptr uint16_t* id, uint32_t S, volatile tt_l1_ptr uint16_t* pos,
                      volatile tt_l1_ptr uint16_t* uq, volatile tt_l1_ptr uint16_t* act, uint32_t K) {""", 1)
    s = s.replace("""    for (uint32_t s = 0; s < S; s++) {
        const uint32_t e = id[s];""", """    for (uint32_t s = 0; s < S; s++) {
        if (act[s / K] == 0) {  // an idle slot's row: its experts are not computed
            continue;
        }
        const uint32_t e = id[s];""", 1)
    assert s.count("act[s / K]") == 1
    open(KD + f, "w").write(s)

s = open(KD + "expert_mm3_reader.cpp").read()
s = s.replace("    constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();\n",
              "    constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();\n"
              "    constexpr auto m_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();\n"
              "    constexpr uint32_t KR = get_compile_time_arg_val(m_a.next_compile_time_args_offset());  // routed per row\n", 1)
s = s.replace("    const uint32_t SG = get_arg_val<uint32_t>(5);\n", "    const uint32_t SG = get_arg_val<uint32_t>(5);\n    const uint32_t m_addr = get_arg_val<uint32_t>(6);\n", 1)
s = s.replace("    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);\n", "    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);\n    const auto msk = TensorAccessor(m_a, m_addr, 64);\n", 1)
s = s.replace("""    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read_barrier();""", """    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);  // the active-row mask (32 x uint16)
    noc_async_read_barrier();""", 1)
s = s.replace("""reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq);""", """reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536), KR);""", 1)
assert s.count("pi + 1536") == 2
open(KD + "expert_mm3_reader.cpp", "w").write(s)

s = open(KD + "expert_mm3_writer.cpp").read()
s = s.replace("    constexpr auto o_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();\n",
              "    constexpr auto o_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();\n"
              "    constexpr auto m_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();\n"
              "    constexpr uint32_t KR = get_compile_time_arg_val(m_a.next_compile_time_args_offset());\n", 1)
s = s.replace("    const uint32_t SG = get_arg_val<uint32_t>(5);\n", "    const uint32_t SG = get_arg_val<uint32_t>(5);\n    const uint32_t m_addr = get_arg_val<uint32_t>(6);\n", 1)
s = s.replace("    const auto o = TensorAccessor(o_a, o_addr, tb);\n", "    const auto o = TensorAccessor(o_a, o_addr, tb);\n    const auto msk = TensorAccessor(m_a, m_addr, 64);\n", 1)
s = s.replace("""    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read_barrier();""", """    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);
    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);
    noc_async_read_barrier();""", 1)
s = s.replace("""reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq);""", """reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512), uq,
                             reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536), KR);""", 1)
assert s.count("pi + 1536") == 2
open(KD + "expert_mm3_writer.cpp", "w").write(s)

# ---- combine b3 reader
s = open(KD + "moe_combine_b3_reader.cpp").read()
s = s.replace("    constexpr auto i_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();\n",
              "    constexpr auto i_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();\n"
              "    constexpr auto m_a = TensorAccessorArgs<i_a.next_compile_time_args_offset()>();\n", 1)
s = s.replace("    const uint32_t i_addr = get_arg_val<uint32_t>(6);\n", "    const uint32_t i_addr = get_arg_val<uint32_t>(6);\n    const uint32_t m_addr = get_arg_val<uint32_t>(7);\n", 1)
s = s.replace("    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);\n", "    const auto ids = TensorAccessor(i_a, i_addr, 2 * S);\n    const auto msk = TensorAccessor(m_a, m_addr, 64);\n", 1)
s = s.replace("    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);\n", "    noc_async_read(ids.get_noc_addr(0, 0), pi, 2 * S);\n    noc_async_read(msk.get_noc_addr(0, 0), pi + 1536, 64);\n", 1)
s = s.replace("""    volatile tt_l1_ptr uint16_t* pos = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512);
""", """    volatile tt_l1_ptr uint16_t* pos = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 512);
    volatile tt_l1_ptr uint16_t* act = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pi + 1536);
""", 1)
s = s.replace("""    for (uint32_t s = 0; s < S; s++) {
        if (pos[id[s]] == 0xFFFF) {""", """    for (uint32_t s = 0; s < S; s++) {
        if (act[s / K] != 0 && pos[id[s]] == 0xFFFF) {""", 1)
s = s.replace("""    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t j = 0; j < K; j++) {
            dst[pos[id[b * K + j]] * (tb / 2) + elem(b, 0)] = src[elem(b, j)];
        }
    }""", """    for (uint32_t b = 0; b < B; b++) {
        if (act[b] == 0) {  // an idle slot's row: shared expert only (its output is dropped)
            continue;
        }
        for (uint32_t j = 0; j < K; j++) {
            dst[pos[id[b * K + j]] * (tb / 2) + elem(b, 0)] = src[elem(b, j)];
        }
    }""", 1)
assert s.count("act[") == 2, s.count("act[")
s = s.replace("// to expert uq[u]; a row routes to an expert at most once); then per hidden tile c the U down tiles and shared tile c.",
              "// to expert uq[u]; a row routes to an expert at most once); then per hidden tile c the U down tiles and shared tile c.\n"
              "// Rows with mask[b] = 0 (idle continuous-batching slots) are left out of the unique list and get the shared expert only.", 1)
open(KD + "moe_combine_b3_reader.cpp", "w").write(s)

# ---- python
p = "tt/moe_fused_decode.py"
s = open(p).read()
s = s.replace("""        ug = _BATCH["emb_gu"](x4, w.gate_up_proj, idx, B * k, True)  # [1, NSLOT, 32, 2I]""",
              """        msk = active_mask(mesh, B)
        ug = _BATCH["emb_gu"](x4, w.gate_up_proj, idx, B * k, True, msk, k)  # [1, NSLOT, 32, 2I]""", 1)
s = s.replace("""        down = _BATCH["emb_dn"](act, w.down_proj, idx, B * k, False)  # [1, NSLOT, 32, H]""",
              """        down = _BATCH["emb_dn"](act, w.down_proj, idx, B * k, False, msk, k)  # [1, NSLOT, 32, H]""", 1)
s = s.replace("""        out = _BATCH["mc3"](down, v, sh, g, idx)""", """        out = _BATCH["mc3"](down, v, sh, g, idx, msk)""", 1)
s = s.replace('''class ExpertMMB:''', '''def active_mask(mesh, B):
    """The active-row mask of the dedup path ([1, 32] uint16 ROW_MAJOR in DRAM, persistent per B, all ones until
    set_active_rows): rows with 0 are left out of the unique experts."""
    m = _BATCH.setdefault("mask", {})
    if B not in m:
        m[B] = ttnn.from_torch(torch.ones(1, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    return m[B]


def set_active_rows(mesh, B, active):
    """port10 DRAMA3_MOE_DEDUP_MASK (host/cbatch.py): mark the continuous-batching slots that hold a job (outside any
    trace: the traced kernels read the mask buffer at its fixed address)."""
    m = torch.zeros(1, 32, dtype=torch.int32)
    m[0, : len(active)] = torch.tensor([1 if a else 0 for a in active], dtype=torch.int32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT), active_mask(mesh, B))


class ExpertMMB:''', 1)
s = s.replace("""    def __call__(self, a, w, ids, S, shared_a):
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32""", """    def __call__(self, a, w, ids, S, shared_a, msk, k):
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32""", 1)
s = s.replace("""            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), n0, s0, SG]
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), n0, s0, SG]""",
              """            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), n0, s0, SG, msk.buffer_address()]
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), n0, s0, SG, msk.buffer_address()]""", 1)
s = s.replace("""            kd("expert_mm3_reader.cpp", [KT, NT, KB, int(shared_a), S, D, C, NTC] + acc(a, w, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd("expert_mm3_writer.cpp", [KT, NT, KB, S, D, C, NTC] + acc(w, ids, out), rw, ttnn.WriterConfigDescriptor()),""",
              """            kd("expert_mm3_reader.cpp", [KT, NT, KB, int(shared_a), S, D, C, NTC] + acc(a, w, ids, msk) + [k], rr, ttnn.ReaderConfigDescriptor()),
            kd("expert_mm3_writer.cpp", [KT, NT, KB, S, D, C, NTC] + acc(w, ids, out, msk) + [k], rw, ttnn.WriterConfigDescriptor()),""", 1)
s = s.replace("""        ttnn.generic_op([a, w, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


class MoECombineB3:""", """        ttnn.generic_op([a, w, ids, msk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


class MoECombineB3:""", 1)
s = s.replace("""    def __call__(self, down, v, shared, g, ids):
        B = v.shape[-2]""", """    def __call__(self, down, v, shared, g, ids, msk):
        B = v.shape[-2]""", 1)
s = s.replace("""        for t in (down, v, shared, g, ids):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())""", """        for t in (down, v, shared, g, ids, msk):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())""", 1)
s = s.replace("""g.buffer_address(), ids.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]""", """g.buffer_address(), ids.buffer_address(), msk.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]""", 1)
s = s.replace("""        ttnn.generic_op([down, v, shared, g, ids, out], ttnn.ProgramDescriptor""", """        ttnn.generic_op([down, v, shared, g, ids, msk, out], ttnn.ProgramDescriptor""", 1)
assert s.count("msk") >= 14, s.count("msk")
open(p, "w").write(s)
print("ok")
