p = "tt/attn_fused_decode.py"
s = open(p).read()
old = """    m = self.mesh
    g = m.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(8 * B)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
"""
new = """    m = self.mesh
    g = m.compute_with_storage_grid_size()
    if 8 * B > g.x * g.y:
        return _batch_post2(self, attn, gpk, B)
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(8 * B)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
"""
assert old in s
s = s.replace(old, new, 1)
anchor = "class AttnPostFused:"
add = '''def _batch_post2(self, attn, gpk, B):
    """port10: _batch_post when 8 B cores do not fit the grid (B <= 32): 8 G cores, G = cores // 8 row groups, core
    (j, g0) runs rows g0, g0 + G, .. (kernels/attn_post_b2_*.cpp)."""
    m = self.mesh
    g = m.compute_with_storage_grid_size()
    assert B <= 32
    G = (g.x * g.y) // 8
    cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(8 * G)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.L1_MEMORY_CONFIG)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        g0, j = i // 8, i % 8
        rr[c.x][c.y] = [j, attn.buffer_address(), gpk.buffer_address(), g0, G, B]
        rw[c.x][c.y] = [j, out.buffer_address(), g0, G, B]
        rc[c.x][c.y] = [len(range(g0, B, G))]
    kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct, runtime_args=rt, config=cfg,
    )
    kernels = [
        kd("attn_post_b2_reader.cpp", acc(attn) + acc(gpk), rr, ttnn.ReaderConfigDescriptor()),
        kd("attn_post_b2_writer.cpp", [16] + acc(out), rw, ttnn.WriterConfigDescriptor()),
        kd("attn_post_b2_compute.cpp", [], rc,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
    ]
    cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=crs,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
           for i, n in ((0, 2), (1, 2), (3, 1), (16, 2))]
    ttnn.generic_op([attn, gpk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


'''
s = s.replace(anchor, add + anchor, 1)
open(p, "w").write(s)
print("ok")
