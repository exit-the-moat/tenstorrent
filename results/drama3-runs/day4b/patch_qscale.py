p = "/opt/ep0/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py"
s = open(p).read()
old = '''    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)
    q = ttnn.multiply(q, scale if scale is not None else K**-0.5, memory_config=L1)
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=L1)'''
assert s.count(old) == 1
s = s.replace(old, '''    import os

    if os.environ.get("QWEN36_GDN_LEAN_QSCALE", "0") == "1":
        # q's l2 norm (rms_norm * K^-0.5) and the K^-0.5 scale in ONE multiply (by 1 / K, exact in bf16)
        q = ttnn.rms_norm(q, epsilon=1e-6 / K)
        q = ttnn.multiply(q, K**-0.5 * (scale if scale is not None else K**-0.5), memory_config=L1)
    else:
        q = l2_norm_ttnn(q, dim=-1)
        q = ttnn.multiply(q, scale if scale is not None else K**-0.5, memory_config=L1)
    k = l2_norm_ttnn(k, dim=-1)
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=L1)''')
open(p, "w").write(s)
print("ok")
