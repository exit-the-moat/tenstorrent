"""port10: the shared expert as a 9th slot of ExpertMM v2 (DRAMA3_MOE_SHARED_IN_EMM=1): expert_mm5_reader/writer,
moe_combine_s_reader, the Python wiring."""
KD = "kernels/"
note = ("// drama3 port10: expert_mm5 = expert_mm2 (ExpertMM v2) with one extra slot NSLOT - 1 that streams the shared\n"
        "// expert's weights (a second weight tensor, expert 0 of it) instead of a routed expert: the shared expert's\n"
        "// weight stream runs inside the routed experts' program (DRAMA3_MOE_SHARED_IN_EMM=1). ids holds NSLOT - 1 routed ids.\n")

for side in ("reader", "writer"):
    s = open(KD + f"expert_mm2_{side}.cpp", encoding="latin-1").read()
    s = note + s
    # the shared weight accessor right after the routed weight accessor
    s = s.replace("constexpr auto i_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();",
                  "constexpr auto ws_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();\n"
                  "    constexpr auto i_a = TensorAccessorArgs<ws_a.next_compile_time_args_offset()>();", 1)
    assert "ws_a" in s, side
    if side == "reader":
        s = s.replace("    const uint32_t nj = get_arg_val<uint32_t>(4);\n",
                      "    const uint32_t nj = get_arg_val<uint32_t>(4);\n    const uint32_t ws_addr = get_arg_val<uint32_t>(5);\n", 1)
    else:
        s = s.replace("    const uint32_t nj = get_arg_val<uint32_t>(4);\n",
                      "    const uint32_t nj = get_arg_val<uint32_t>(4);\n    const uint32_t ws_addr = get_arg_val<uint32_t>(5);\n", 1)
    assert "ws_addr = get_arg_val" in s, side
    s = s.replace("    const auto w = TensorAccessor(w_a, w_addr, tw);\n",
                  "    const auto w = TensorAccessor(w_a, w_addr, tw);\n    const auto ws = TensorAccessor(ws_a, ws_addr, tw);\n", 1)
    assert "const auto ws =" in s, side
    # the routed ids: NSLOT - 1 of them
    s = s.replace("const auto ids = TensorAccessor(i_a, i_addr, 2 * NSLOT);", "const auto ids = TensorAccessor(i_a, i_addr, 2 * (NSLOT - 1));", 1)
    s = s.replace("noc_async_read(ids.get_noc_addr(0, 0), ci.get_write_ptr(), 2 * NSLOT);",
                  "noc_async_read(ids.get_noc_addr(0, 0), ci.get_write_ptr(), 2 * (NSLOT - 1));", 1)
    assert s.count("2 * (NSLOT - 1)") == 2, side
    s = s.replace("            const uint32_t e = id[j];\n",
                  "            const bool shs = j == NSLOT - 1;  // the shared expert's slot\n            const uint32_t e = shs ? 0 : id[j];\n", 1)
    assert "shs ?" in s, side
    old = "noc_async_read(w.get_noc_addr((e * KT + k0 + 2 * k"
    assert s.count(old) == 1, side
    s = s.replace(old, "noc_async_read((shs ? ws : w).get_noc_addr((e * KT + k0 + 2 * k", 1)
    open(KD + f"expert_mm5_{side}.cpp", "w", encoding="latin-1").write(s)

c = open(KD + "moe_combine_reader.cpp", encoding="latin-1").read()
c = ("// drama3 port10: moe_combine_s = moe_combine with the shared expert's rows at page SOFF + c of the shared tensor\n"
     "// (DRAMA3_MOE_SHARED_IN_EMM=1: slot k of the ExpertMM down output).\n") + c
c = c.replace("    constexpr uint32_t HT = get_compile_time_arg_val(1);  // hidden tiles\n",
              "    constexpr uint32_t HT = get_compile_time_arg_val(1);  // hidden tiles\n    constexpr uint32_t SOFF = K * HT;\n", 1)
assert c.count("{.page_id = c}") == 1
c = c.replace("{.page_id = c}", "{.page_id = SOFF + c}", 1)
open(KD + "moe_combine_s_reader.cpp", "w", encoding="latin-1").write(c)

p = "tt/moe_fused_decode.py"
s = open(p).read()
# MoECombine: optional shared-in-down
old = """    def __call__(self, down, v, shared, g):
        L1 = ttnn.L1_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)"""
new = """    def __call__(self, down, v, shared, g, shared_in_down=False):
        \"\"\"shared_in_down (port10): shared is None, the shared expert's rows are slot k of down.\"\"\"
        L1 = ttnn.L1_MEMORY_CONFIG
        shared = down if shared_in_down else shared
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = """            kd("moe_combine_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),"""
new = """            kd("moe_combine_s_reader.cpp" if shared_in_down else "moe_combine_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = """        ttnn.generic_op([down, v, shared, g, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


_EXPERT_MM = {}"""
new = """        ttnn.generic_op([down, v, g, out] if shared_in_down else [down, v, shared, g, out],
                        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


_EXPERT_MM = {}


def shared_in_emm():
    \"\"\"DRAMA3_MOE_SHARED_IN_EMM=1 (port10, opt-in, with DRAMA3_MOE_EXPERT_MM=1 V=2 and the fused combine / SwiGLU):
    the shared expert's [w3 | w1] and w2 stream as a 9th slot of the routed ExpertMM programs (one program per
    matmul for routed + shared, no separate shared-expert matmuls), MoECombine reads its rows from slot k.\"\"\"
    return (os.environ.get("DRAMA3_MOE_SHARED_IN_EMM", "0") == "1" and os.environ.get("DRAMA3_MOE_EXPERT_MM", "0") == "1"
            and os.environ.get("DRAMA3_MOE_EXPERT_MM_V", "1") == "2")


def _down_shared(x4, idx, weights, config, sf):
    \"\"\"port10: routed + shared expert through ExpertMM (expert_mm5): [1, k + 1, 1, H], slot k = the shared expert.\"\"\"
    mesh = x4.device()
    if "gu5" not in _EXPERT_MM:
        kb = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_KB", "16"))
        _EXPERT_MM["gu5"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_CORES_GU", "64")), kb)
        _EXPERT_MM["dn5"] = ExpertMM(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_CORES_DN", "64")), kb)
        _EXPERT_MM["sw5"] = Swiglu(mesh, int(os.environ.get("DRAMA3_MOE_SHEMM_SWIGLU_CORES", "48")))
    k = config.top_k
    w = weights
    ws_gu = ttnn.reshape(sf.w_gu, (1, 1, sf.w_gu.shape[-2], sf.w_gu.shape[-1]))
    ws_dn = ttnn.reshape(sf.w2, (1, 1, sf.w2.shape[-2], sf.w2.shape[-1]))
    up_gate = _EXPERT_MM["gu5"](x4, w.gate_up_proj, idx, k + 1, True, ws=ws_gu)
    act = _EXPERT_MM["sw5"](up_gate)
    up_gate.deallocate(True)
    down = _EXPERT_MM["dn5"](act, w.down_proj, idx, k + 1, False, ws=ws_dn)
    act.deallocate(True)
    return down"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
# the one-token forward
old = """            vals, idx = _m.router.topk(x4)
            down = _down_indexed(x4, idx, _m.experts.weights, _m.config, _m.experts.decode_sparsity, sw_routed)
            g = ttnn.linear(x3, _m.shared.gate_weight)"""
new = """            vals, idx = _m.router.topk(x4)
            if _sf is not None and shared_in_emm() and idx.layout == ttnn.ROW_MAJOR_LAYOUT:
                down = _down_shared(x4, idx, _m.experts.weights, _m.config, _sf)
                g = ttnn.linear(x3, _m.shared.gate_weight)
                out = mc(down, vals, None, g, shared_in_down=True)
                for t in (down, g, vals, idx):
                    ttnn.deallocate(t)
                return ttnn.reshape(out, tuple(x.shape))
            down = _down_indexed(x4, idx, _m.experts.weights, _m.config, _m.experts.decode_sparsity, sw_routed)
            g = ttnn.linear(x3, _m.shared.gate_weight)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
# ExpertMM: optional shared weight tensor -> the v5 kernels
old = """    def __call__(self, a, w, ids, nslot, shared_a):
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32"""
new = """    def __call__(self, a, w, ids, nslot, shared_a, ws=None):
        \"\"\"ws (port10, v2 only): the shared expert's weights [1, 1, K, N] as slot nslot - 1 (kernels/expert_mm5_*).\"\"\"
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = """            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), j0, nj]
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), j0, nj]"""
new = """            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), j0, nj] + ([ws.buffer_address()] if ws is not None else [])
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), j0, nj] + ([ws.buffer_address()] if ws is not None else [])"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = """        sfx = "2" if v2 else ""
        kernels = [
            kd(f"expert_mm{sfx}_reader.cpp", [KT, NT, KB, int(shared_a), nslot] + dc + acc(a, w, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd(f"expert_mm{sfx}_writer.cpp", [KT, NT, KB, nslot] + dc + acc(w, ids, out), rw, ttnn.WriterConfigDescriptor()),"""
new = """        sfx = "2" if v2 else ""
        if ws is not None:
            assert v2 and ws.dtype == w.dtype and tuple(ws.shape[-2:]) == tuple(w.shape[-2:]), (ws.shape, w.shape, ws.dtype, w.dtype)
        rsrc, wsrc = (f"expert_mm{sfx}_reader.cpp", f"expert_mm{sfx}_writer.cpp") if ws is None else ("expert_mm5_reader.cpp", "expert_mm5_writer.cpp")
        wl = [w] if ws is None else [w, ws]
        kernels = [
            kd(rsrc, [KT, NT, KB, int(shared_a), nslot] + dc + acc(a, *wl, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd(wsrc, [KT, NT, KB, nslot] + dc + acc(*wl, ids, out), rw, ttnn.WriterConfigDescriptor()),"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = """        ttnn.generic_op([a, w, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out"""
assert s.count(old) == 1
s = s.replace(old, """        ttnn.generic_op([a, *wl, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out""", 1)
open(p, "w").write(s)
print("ok")
