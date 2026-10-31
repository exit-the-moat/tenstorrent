"""Day 4b: GDN one-token output gate without the overflow clips, silu fused (opt-in QWEN36_GDN_DECODE_NOCLIP=1)."""
p = "/opt/ep0/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py"
s = open(p).read()
old = '''        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mc)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mc)

    o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)'''
assert s.count(old) == 1
s = s.replace(old, '''        if T == 1 and _os.environ.get("QWEN36_GDN_DECODE_NOCLIP", "0") == "1":
            # one token: the +-1e4 clips (against overflow at long T) never bind; silu fused into the multiply
            o = ttnn.rms_norm(o, weight=o_norm_weight, epsilon=norm_eps, memory_config=mc)
            o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        else:
            o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mc)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mc)

    if not (T == 1 and _os.environ.get("QWEN36_GDN_DECODE_NOCLIP", "0") == "1"):
        o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)''')
if "import os as _os" not in s:
    s = s.replace("import ttnn\n", "import os as _os\n\nimport ttnn\n", 1)
open(p, "w").write(s)
print("ok")
