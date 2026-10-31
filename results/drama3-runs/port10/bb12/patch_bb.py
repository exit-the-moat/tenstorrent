import os
import sys

p = sys.argv[1]
s = open(p).read()

a_old = """        self.trace_id = None
        self.dec_out = None
        self.build_s = time.time() - t0
"""
a_new = """        self.trace_id = None
        self.dec_out = None
        # DRAMA3_BB_PREFETCH=1 (port10 bb12, opt-in): the one-token decode body streams its dense weights (GDN mega
        # in-projection + o_proj, attention [q|gate | k | v] + o_proj: 80 per frame) through the DRAM-core tensor
        # prefetcher (tt/weight_prefetch.py, group "backbone"; one global CB shared with the tower). Needs the fused
        # GDN decode and the merged attention qkv; prefill and batch mode (host/batch, host/cbatch) unchanged.
        self.pf = None
        self._pf_capture = False
        if os.environ.get("DRAMA3_BB_PREFETCH", "0") == "1":
            from models.demos.blackhole.drama3.tt import weight_prefetch

            if not weight_prefetch.batch_main():
                t1 = time.time()
                self.pf = weight_prefetch.get(mesh).group("backbone")
                for i, layer in enumerate(self.layers):
                    if layer.is_full_attention:
                        qkv = weight_prefetch.attn_qkv(layer.attention)
                        assert qkv is not None, "DRAMA3_BB_PREFETCH needs DRAMA3_ATTN_FUSED_PRE=1 + DRAMA3_ATTN_QKV_MERGED=1"
                        self.pf.add(qkv, f"L{i}.qkv")
                        self.pf.add(layer.attention.weights.o_proj, f"L{i}.o_proj")
                    else:
                        assert getattr(layer.attention, "fused_decode", None) is not None, "DRAMA3_BB_PREFETCH needs DRAMA3_GDN_FUSED_DECODE=all"
                        w = layer.attention.weights
                        self.pf.add(w.mega_fused_weight, f"L{i}.in_proj")
                        self.pf.add(w.o_proj_weight, f"L{i}.o_proj")
                log(f"backbone prefetch: {len(self.pf.order)} weights, {self.pf.dram_bytes() / 1e9:.2f} GB copies in {time.time() - t1:.0f}s")
        self.build_s = time.time() - t0
"""
assert s.count(a_old) == 1
s = s.replace(a_old, a_new)

b_old = """    def _decode_body(self, collect=False):
        if self.addnorm is not None and not collect:
            return self._decode_body_addnorm()
"""
b_new = """    def _decode_body(self, collect=False):
        if self.pf is not None:  # DRAMA3_BB_PREFETCH=1: one request per body, captured with the trace
            with self.pf.frame(capture=self._pf_capture):
                return self._decode_body_inner(collect)
        return self._decode_body_inner(collect)

    def _decode_body_inner(self, collect=False):
        if self.addnorm is not None and not collect:
            return self._decode_body_addnorm()
"""
assert s.count(b_old) == 1
s = s.replace(b_old, b_new)

c_old = """        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.dec_out = self._decode_body()[:2]
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
"""
c_new = """        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._pf_capture = True  # DRAMA3_BB_PREFETCH: the request is captured too (every replay re-sends it)
        try:
            self.dec_out = self._decode_body()[:2]
        finally:
            self._pf_capture = False
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
"""
assert s.count(c_old) == 1
s = s.replace(c_old, c_new)

tmp = p + ".tmp"
open(tmp, "w").write(s)
os.replace(tmp, p)
print("patched", p)
