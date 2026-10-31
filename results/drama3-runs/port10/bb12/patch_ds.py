import os

p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/device_sampler.py"
s = open(p).read()

a_old = """        c = torch.zeros(K)
        c[0], c[top_k:] = -1e4, 1e4
        self.keep_bias = const(c.reshape(1, 1, 1, K))
"""
a_new = """        c = torch.zeros(K)
        c[0], c[top_k:] = -1e4, 1e4
        self.keep_bias = const(c.reshape(1, 1, 1, K))
        # DRAMA3_SAMPLER_FUSED=1 (port10 bb12, opt-in): one row's draw as ONE program (FusedDraw) with the lean math
        self.fused = FusedDraw(mesh, n_codes, top_k, top_p) if os.environ.get("DRAMA3_SAMPLER_FUSED", "0") == "1" else None
"""
assert s.count(a_old) == 1
s = s.replace(a_old, a_new)

b_old = """        x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        vals, idx = ttnn.topk(x, K, dim=-1, largest=True, sorted=True)
"""
b_new = """        x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        if self.fused is not None and x.shape[2] == 1 and x.shape[-1] == self.fused.n:
            return self.fused(x, noise)
        vals, idx = ttnn.topk(x, K, dim=-1, largest=True, sorted=True)
"""
assert s.count(b_old) == 1
s = s.replace(b_old, b_new)

s += '''

KDIR = "models/demos/blackhole/drama3/kernels"


class FusedDraw:
    """DRAMA3_SAMPLER_FUSED=1 (port10 bb12): the lean draw of one row in one program on one core
    (kernels/sampler_fused_reader.cpp + sampler_fused_compute.cpp): top-32 on the RISC (ties: lower code), E = exp(x - m)
    over all logits on the SFPU (the ops ttnn.subtract / ttnn.exp run), z = sum(E) as an exact fixed-point sum rounded
    once, then p / cum / keep / e / noise / argmax in fp32 in rank order. The codes match the lean sampler's except
    where fp32 rounding of z, p or cum differs at a threshold (tests/check_sampler_fused.py counts them).
    Core: the last grid column, row 0 (outside the prefetcher's receiver cores (0..7, 0..7))."""

    def __init__(self, mesh, n_codes, top_k, top_p):
        import struct

        assert n_codes == 1024 and top_k <= K
        self.mesh, self.n, self.top_k = mesh, n_codes, top_k
        self.top_p_bits = struct.unpack("<I", struct.pack("<f", float(top_p)))[0]
        g = mesh.compute_with_storage_grid_size()
        c = ttnn.CoreCoord(int(os.environ.get("DRAMA3_SAMPLER_FUSED_CORE_X", g.x - 1)), 0)
        self.core, self.crs = c, ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
        self.ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.ccfg.unpack_to_dest_mode = modes

    def __call__(self, logits, noise):
        """logits [1, 1, 1, n] fp32 TILE, noise [1, 1, 1, K] fp32 TILE -> (code [1, 1, 1, 1] fp32, one-hot [1, 1, 1, n] bf16)."""
        code = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        ct = [self.n, self.top_k, self.top_p_bits]
        for t in (logits, noise, code, oh):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [logits.buffer_address(), noise.buffer_address(), code.buffer_address(), oh.buffer_address()]
        kd = lambda src, ct_, rt_, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg,
        )
        kernels = [
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_fused_compute.cpp", [], [], self.ccfg),
        ]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [cb(0, f32, 4096, 1), cb(1, f32, 4096, 1), cb(2, f32, 4096, 1), cb(3, f32, 4096, 1), cb(4, bf, 2048, 2), cb(5, f32, 4096, 1)]
        ttnn.generic_op([logits, noise, code, oh], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return code, oh
'''
open(p + ".tmp", "w").write(s)
os.replace(p + ".tmp", p)
print("ok")
