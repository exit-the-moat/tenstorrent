import os

p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/backbone_paged.py"
s = open(p).read()
old = """                self.pf = weight_prefetch.get(mesh).group("backbone")
                for i, layer in enumerate(self.layers):"""
new = """                self.pf = weight_prefetch.get(mesh).group("backbone")
                # DRAMA3_BB_PREFETCH_ONLY (checks): a comma list of the kinds to stream (attn_qkv, attn_o, gdn_in, gdn_o)
                env = os.environ.get("DRAMA3_BB_PREFETCH_ONLY", "")
                only = set(filter(None, env.split(","))) or {"attn_qkv", "attn_o", "gdn_in", "gdn_o"}
                for i, layer in enumerate(self.layers):"""
assert s.count(old) == 1
s = s.replace(old, new)
for a, b in [
    ('                        self.pf.add(qkv, f"L{i}.qkv")\n', '                        if "attn_qkv" in only:\n                            self.pf.add(qkv, f"L{i}.qkv")\n'),
    ('                        self.pf.add(layer.attention.weights.o_proj, f"L{i}.o_proj")\n', '                        if "attn_o" in only:\n                            self.pf.add(layer.attention.weights.o_proj, f"L{i}.o_proj")\n'),
    ('                        self.pf.add(w.mega_fused_weight, f"L{i}.in_proj")\n', '                        if "gdn_in" in only:\n                            self.pf.add(w.mega_fused_weight, f"L{i}.in_proj")\n'),
    ('                        self.pf.add(w.o_proj_weight, f"L{i}.o_proj")\n', '                        if "gdn_o" in only:\n                            self.pf.add(w.o_proj_weight, f"L{i}.o_proj")\n'),
]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p + ".tmp", "w").write(s)
os.replace(p + ".tmp", p)
print("ok")
