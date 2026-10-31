import os
p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/device_sampler.py"
s = open(p).read()
old = """        self.fused = FusedDraw(mesh, n_codes, top_k, top_p) if os.environ.get("DRAMA3_SAMPLER_FUSED", "0") == "1" else None
"""
new = """        self.fused = None
        if os.environ.get("DRAMA3_SAMPLER_FUSED", "0") == "1":
            nc = int(os.environ.get("DRAMA3_SAMPLER_FUSED_CORES", "1"))
            self.fused = FusedDraw(mesh, n_codes, top_k, top_p) if nc == 1 else FusedDrawMC(mesh, n_codes, top_k, top_p, nc)
"""
assert s.count(old) == 1; s = s.replace(old, new)
s += '''

class FusedDrawMC(FusedDraw):
    """DRAMA3_SAMPLER_FUSED_CORES=NC (2, 4 or 8; with DRAMA3_SAMPLER_FUSED=1): the fused draw over NC cores of the last grid
    column (rows 0 ..; kernels/sampler_mc_*.cpp): core j takes the top-32 and the exp sum of logits [j N / NC, ..),
    core 0 merges and draws. z = sum_j exp(m_j - m) * sum_j (instead of one sum over all exp(x - m)): the codes can
    differ from the one-core kernel's where fp32 rounding of z flips a threshold."""

    def __init__(self, mesh, n_codes, top_k, top_p, nc):
        super().__init__(mesh, n_codes, top_k, top_p)
        assert nc in (2, 4, 8)
        self.nc = nc
        g = mesh.compute_with_storage_grid_size()
        x = self.core.x
        self.cores = [ttnn.CoreCoord(x, r) for r in range(nc)]
        assert nc <= g.y
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])
        cn = mesh.worker_core_from_logical_core(self.cores[0])
        self.cnoc = (cn.x, cn.y)
        modes = list(self.ccfg.unpack_to_dest_mode)
        for i in (0, 1, 6, 7, 8, 11, 12):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.ccfg.unpack_to_dest_mode = modes

    def __call__(self, logits, noise):
        code = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        ct = [self.n, self.top_k, self.top_p_bits, self.nc]
        for t in (logits, noise, code, oh):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [logits.buffer_address(), noise.buffer_address(), code.buffer_address(), oh.buffer_address()]
        for j, c in enumerate(self.cores):
            rt[c.x][c.y] = addrs + [j, self.cnoc[0], self.cnoc[1]]
            rc[c.x][c.y] = [j]
        kd = lambda src, ct_, rt_, cfg, defines=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg, defines=list(defines),
        )
        kernels = [
            kd("sampler_mc_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_mc_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")]),
            kd("sampler_mc_compute.cpp", [], rc, self.ccfg),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(6)]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [cb(i, f32, 4096, 1) for i in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12)] + [cb(4, bf, 2048, 2), cb(5, f32, 8192, 1)]
        ttnn.generic_op([logits, noise, code, oh], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return code, oh
'''
open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p); print("ok")
