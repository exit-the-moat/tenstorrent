import os

os.chdir("/opt/ep0/tt-metal/models/demos/blackhole")

p = "qwen36/tt/moe/weights.py"
s = open(p).read()
anchor = "def load_expert_weights("
add = '''def gate_up_dtype_for_cache(tensor_cache_path, default=ttnn.bfloat4_b):
    """Opt-in per-layer gate/up precision (single-device experiments): QWEN36_EXPERT_GU_BFP8_LAYERS = "all" or a
    comma list of layer ids -> bfloat8_b for those layers. The layer id is read from the per-layer cache dir name
    (``layers.N``); unset (default) or no cache path -> ``default`` (bfloat4_b)."""
    spec = os.environ.get("QWEN36_EXPERT_GU_BFP8_LAYERS", "").strip()
    if not spec or tensor_cache_path is None:
        return default
    name = Path(tensor_cache_path).name
    if not name.startswith("layers."):
        return default
    if spec == "all" or int(name.split(".")[1]) in {int(x) for x in spec.split(",") if x}:
        return ttnn.bfloat8_b
    return default


'''
assert "gate_up_dtype_for_cache" not in s
s = s.replace(anchor, add + anchor, 1)
s = s.replace("from dataclasses import dataclass\n", "import os\nfrom dataclasses import dataclass\nfrom pathlib import Path\n", 1)
open(p, "w").write(s)

p = "qwen36/tt/moe/experts.py"
s = open(p).read()
s = s.replace("from .weights import load_expert_weights", "from .weights import gate_up_dtype_for_cache, load_expert_weights", 1)
o = "self.weights = load_expert_weights(mesh_device, config, state_dict, tensor_cache_path)"
assert o in s
s = s.replace(
    o,
    "self.weights = load_expert_weights(\n"
    "            mesh_device, config, state_dict, tensor_cache_path, gate_up_dtype=gate_up_dtype_for_cache(tensor_cache_path)\n"
    "        )",
)
open(p, "w").write(s)

p = "drama3/tt/device_check.py"
s = open(p).read()
o = '''    warm = all(any((cache / f"layers.{i}").glob(f"moe.experts.{n}_*")) for n in ("gate_proj", "up_proj", "down_proj"))'''
assert o in s
n = '''    # Day 5: the gate/up dtype is per layer (QWEN36_EXPERT_GU_BFP8_LAYERS); a BFP8 layer needs its BFP8 files
    from models.demos.blackhole.qwen36.tt.moe.weights import gate_up_dtype_for_cache

    gu = "BFLOAT8_B" if gate_up_dtype_for_cache(cache / f"layers.{i}") == ttnn.bfloat8_b else "BFLOAT4_B"
    tags = {"gate_proj": gu, "up_proj": gu, "down_proj": "BFLOAT8_B"}
    warm = all(any((cache / f"layers.{i}").glob(f"moe.experts.{n}_dtype_{d}_*")) for n, d in tags.items())'''
s = s.replace(o, n)
open(p, "w").write(s)
print("PATCHED")
