"""ep1 patch v5 on top of gdn_tp_fir_dram.py: QWEN36_GDN_UNFUSED_RS=1 replaces the fused out-proj matmul + reduce-scatter
(prefill) with the existing unfused path (row matmul + tt_all_reduce, which is a reduce-scatter on a 1xN mesh), with the
cross-device sum kept in fp32 like the fused op. Usage: patch_v5.py <in> <out>"""
import sys

s = open(sys.argv[1]).read()
a = """        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1"""
b = """        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1
        # ep1 patch: QWEN36_GDN_UNFUSED_RS=1 -> separate matmul + fp32 reduce-scatter (the fused op puts its RS workers
        # on core row 10 at nd=2, beyond the 10-row p150 grid)
        if os.environ.get("QWEN36_GDN_UNFUSED_RS") == "1":
            self._fuse_out_mmrs_prefill = False"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )"""
b = """        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        _f32 = os.environ.get("QWEN36_GDN_UNFUSED_RS") == "1"  # ep1 patch: cross-device sum in fp32, as the fused op
        if _f32:
            partial = ttnn.typecast(partial, ttnn.float32)
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"dtype": ttnn.float32} if _f32 else {}),
        )"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
open(sys.argv[2], "w").write(s)
print("patched")
