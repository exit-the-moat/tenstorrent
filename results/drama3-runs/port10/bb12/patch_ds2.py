import os
p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/device_sampler.py"
s = open(p).read()
old = """        kd = lambda src, ct_, rt_, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg,
        )
        kernels = [
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_fused_compute.cpp", [], [], self.ccfg),
        ]"""
new = """        kd = lambda src, ct_, rt_, cfg, defines=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg, defines=list(defines),
        )
        kernels = [  # the reader source runs on both data-movement RISCs (SAMPLER_B: the BRISC half)
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")]),
            kd("sampler_fused_compute.cpp", [], [], self.ccfg),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(4)]"""
assert s.count(old) == 1; s = s.replace(old, new)
old2 = "ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))\n        return code, oh"
new2 = "ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))\n        return code, oh"
assert s.count(old2) == 1; s = s.replace(old2, new2)
open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p); print("ok")
