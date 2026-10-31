p = "/opt/ep0/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py"
s = open(p).read()
old = '''    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    ttnn.deallocate(gate)'''
assert s.count(old) == 1
s = s.replace(old, '''    # Apply sigmoid gate (QWEN36_ATTN_GATE_FUSED=1, one token: the sigmoid as the multiply's input activation)
    if T == 1 and _os.environ.get("QWEN36_ATTN_GATE_FUSED", "0") == "1":
        attn_output = ttnn.multiply(attn_output, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    else:
        gate = ttnn.sigmoid(gate)
        attn_output = ttnn.multiply(attn_output, gate)
    ttnn.deallocate(gate)''')
open(p, "w").write(s)
print("ok")
