import shutil

p = "tt/moe_fused_decode.py"
s = open(p).read()

# 1) forward_batch: the dedup branch right after the router (v = softmax of the top-k scores)
old = """    w = moe.experts.weights
    I = w.intermediate_size_per_device
    ot = ttnn.Tile([32, 32])
    gu_cfg = _build_sparse_matmul_config(B, 2 * I"""
new = """    w = moe.experts.weights
    I = w.intermediate_size_per_device
    if dedup_on():  # port10: the expert matmuls over the unique routed experts of the B rows (ExpertMMB)
        if "emb_gu" not in _BATCH:
            _BATCH["emb_gu"] = ExpertMMB(mesh, int(os.environ.get("DRAMA3_MOE_DEDUP_NG_GU", "8")))
            _BATCH["emb_dn"] = ExpertMMB(mesh, int(os.environ.get("DRAMA3_MOE_DEDUP_NG_DN", "4")))
            _BATCH["mc3"] = MoECombineB3(mesh)
        ug = _BATCH["emb_gu"](x4, w.gate_up_proj, idx, B * k, True)  # [1, NSLOT, 32, 2I]
        sw = _BATCH.get("sw")
        act = sw(ug) if sw is not None else apply_swiglu(ug)
        ug.deallocate(True)
        down = _BATCH["emb_dn"](act, w.down_proj, idx, B * k, False)  # [1, NSLOT, 32, H]
        act.deallocate(True)
        g = ttnn.linear(x3, moe.shared.gate_weight)  # [1, B, 1]
        sh = _shared_rows(shared_fused, x3) if shared_fused is not None else moe.shared.mlp.forward(x3)
        out = _BATCH["mc3"](down, v, sh, g, idx)
        for t in (down, g, sh, v, idx):
            ttnn.deallocate(t)
        return ttnn.reshape(out, tuple(x.shape))
    ot = ttnn.Tile([32, 32])
    gu_cfg = _build_sparse_matmul_config(B, 2 * I"""
assert old in s
s = s.replace(old, new, 1)

# 2) the flag, the classes
anchor = "def _shared_rows(sf, x3):"
add = '''def dedup_on():
    """DRAMA3_MOE_BATCH_DEDUP=1 (port10, opt-in, with DRAMA3_MOE_FUSED_BATCH): batch-mode MoE expert matmuls once per
    unique routed expert of the B rows (all rows in its M = 32 tile) instead of once per (row, slot): ExpertMMB +
    MoECombineB3. The unique list is built on the card from the router's ids at run time (trace-safe)."""
    return os.environ.get("DRAMA3_MOE_BATCH_DEDUP", "0") == "1"


class ExpertMMB:
    """port10 ExpertMM v3 (kernels/expert_mm3_*.cpp): out[u] = A_u @ W[uq[u]] over the U unique experts of the B * K
    routed ids (first-occurrence order, U found on the card), 64 cores as NG column groups x SG slot groups (core
    (n0, s0): NT / NG output tiles of the slots s0, s0 + SG, ..). A: the shared [1, 1, B, K] token rows (gate_up) or
    [1, NSLOT, 32, K] per slot (down). -> [1, NSLOT, 32, N] bf16 TILE in L1 (slots >= U not written)."""

    def __init__(self, mesh, NG, n_cores=64):
        assert n_cores % NG == 0
        self.mesh, self.NG, self.SG = mesh, NG, n_cores // NG
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.kb = int(os.environ.get("DRAMA3_MOE_EXPERT_MM_KB", "16"))
        self.D = int(os.environ.get("DRAMA3_MOE_DEDUP_D", "8"))

    def __call__(self, a, w, ids, S, shared_a):
        KT, NT = w.shape[-2] // 32, w.shape[-1] // 32
        KB = min(self.kb, KT)
        assert KT % KB == 0 and NT % self.NG == 0 and S <= 256
        NTC, SG, D = NT // self.NG, self.SG, self.D
        C = 2 * D
        nslot = S
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, nslot, 32, NT * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        acc = lambda *ts: [x for t in ts for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]  # noqa: E731
        rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            n0, s0 = (i % self.NG) * NTC, i // self.NG
            rr[c.x][c.y] = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), n0, s0, SG]
            rw[c.x][c.y] = [w.buffer_address(), ids.buffer_address(), out.buffer_address(), n0, s0, SG]
            rc[c.x][c.y] = [s0, SG]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        fid = getattr(ttnn.MathFidelity, os.environ.get("QWEN36_DECODE_MATH_FIDELITY", "LoFi"))
        kernels = [
            kd("expert_mm3_reader.cpp", [KT, NT, KB, int(shared_a), S, D, C, NTC] + acc(a, w, ids), rr, ttnn.ReaderConfigDescriptor()),
            kd("expert_mm3_writer.cpp", [KT, NT, KB, S, D, C, NTC] + acc(w, ids, out), rw, ttnn.WriterConfigDescriptor()),
            kd("expert_mm3_compute.cpp", [KT, KB, int(shared_a), NTC], rc,
               ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        wt = str(w.dtype).split(".")[-1]
        max_jobs = -(-nslot // SG) * NTC
        cb = lambda i, n, dt=ttnn.bfloat16, page=2048: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])
        cbs = [
            cb(0, KT if shared_a else 3 * KT),
            cb(1, C * KB // 2, w.dtype, TILE_BYTES[wt]),
            cb(4, C * KB // 2, w.dtype, TILE_BYTES[wt]),
            cb(2, 1),  # ids | pos table | unique list (512 bytes each)
            cb(5, 1),
            cb(6, 1, ttnn.bfloat16, 64),  # the unique count for the compute
            cb(16, max_jobs),
        ]
        ttnn.generic_op([a, w, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


class MoECombineB3:
    """port10 kernels/moe_combine_b3_*.cpp: MoECombineB over ExpertMMB's unique slots: out[b] = sum_j v[b, j] *
    down[pos[id[b k + j]]][row b] + sigmoid(g_b) * shared[b]; down [1, NSLOT, 32, H], ids the router's uint16 row."""

    def __init__(self, mesh, H=2048, k=8, n_cores=None):
        n_cores = n_cores or int(os.environ.get("DRAMA3_MOE_COMBINE_B_CORES", "32"))
        self.mesh, self.H, self.k, self.HT = mesh, H, k, H // 32
        assert self.HT % n_cores == 0
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n_cores)]
        self.per = self.HT // n_cores
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, down, v, shared, g, ids):
        B = v.shape[-2]
        S = B * self.k
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        rct = [self.k, self.HT, B]
        for t in (down, v, shared, g, ids):
            rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        wct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            t0 = i * self.per
            rrt[c.x][c.y] = [t0, self.per, down.buffer_address(), v.buffer_address(), shared.buffer_address(), g.buffer_address(), ids.buffer_address()]
            wrt[c.x][c.y] = [t0, self.per, out.buffer_address()]
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct, runtime_args=rt, config=cfg,
        )
        kernels = [
            kd("moe_combine_b3_reader.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kd("moe_combine_writer.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kd("moe_combine_b3_compute.cpp", [S, self.per], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        sizes = {0: (1, 2048), 1: (S, 2048), 2: (S, 2048), 3: (1, 2048), 4: (1, 2048), 5: (1, 2048), 6: (1, 64), 7: (1, 2048), 16: (2, 2048)}
        cbs = [
            ttnn.CBDescriptor(total_size=page * n, core_ranges=self.crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)])
            for i, (n, page) in sizes.items()
        ]
        ttnn.generic_op([down, v, shared, g, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out


'''
s = s.replace(anchor, add + anchor, 1)
open(p, "w").write(s)
print("ok")
