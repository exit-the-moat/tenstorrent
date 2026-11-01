"""ep1 naive-BF4 patch: QWEN36_NAIVE_BF4=1 loads every 8-bit linear weight (MLP down, attention q/k/v/o, GDN
projections, lm_head) as bfloat4_b instead of bfloat8_b. Default (unset) = unchanged. Activations, typecasts and the KV
cache are not touched. Needs its own TT_CACHE_PATH (cache file names don't carry the dtype).
Usage: patch_bf4.py <src dir with mlp.py attention_tp.py gdn_tp_v5.py model.py> <out dir>"""
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
HDR = ('\nimport os as _os_ep1  # ep1 patch\n'
       '_W8 = ttnn.bfloat4_b if _os_ep1.environ.get("QWEN36_NAIVE_BF4") == "1" else ttnn.bfloat8_b  # ep1 naive BF4\n')


def header(s):
    i = s.index("\nimport ttnn\n") + len("\nimport ttnn\n") - 1
    return s[:i] + HDR + s[i:]


# mlp.py: every bfloat8_b there is a weight dtype (down_proj)
s = (src / "mlp.py").read_text()
n = s.count("ttnn.bfloat8_b")
s = header(s.replace("ttnn.bfloat8_b", "_W8"))
(out / "mlp.py").write_text(s)
print("mlp.py", n)

# attention/tp.py and gdn/tp.py: weight loads are `dtype=ttnn.bfloat8_b` followed by , or newline / )
for name in ("attention_tp.py", "gdn_tp_v5.py"):
    s = (src / name).read_text()
    s, n = re.subn(r"dtype=ttnn\.bfloat8_b(?=\s*,|\s*\n)", "dtype=_W8", s)
    (out / name).write_text(header(s))
    print(name, n)

# model.py: only the lm_head weight
s = (src / "model.py").read_text()
a = "self.lm_head_weight = ttnn.as_tensor(\n            lm_head_weight,\n            dtype=ttnn.bfloat8_b,"
assert s.count(a) == 1
s = header(s.replace(a, a.replace("ttnn.bfloat8_b", "_W8")))
(out / "model.py").write_text(s)
print("model.py 1")
