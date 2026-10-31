"""Day 4b: indexed MoE decode, weight the experts before down_proj (QWEN36_MOE_DECODE_PRESCALE=1)."""
p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/decode.py"
s = open(p).read()
old = '''    down_input = apply_swiglu(up_gate)  # [1, k, 1, I]
    up_gate.deallocate(True)
    down = ttnn.sparse_matmul('''
assert s.count(old) == 1
s = s.replace(old, '''    down_input = apply_swiglu(up_gate)  # [1, k, 1, I]
    up_gate.deallocate(True)
    prescale = os.environ.get("QWEN36_MOE_DECODE_PRESCALE", "0") == "1"
    if prescale:  # w_i * (a_i @ W_i) = (w_i * a_i) @ W_i: the routing weights go in before down_proj
        down_input = ttnn.mul(down_input, ttnn.reshape(top_k_values, (1, k, 1, 1)))
    down = ttnn.sparse_matmul(''')
old = '''    if down.shape[-2] != 1:  # the compact-A output keeps the tile-padded M of A (32); row 0 is the token
        down = ttnn.slice(down, [0, 0, 0, 0], [1, k, 1, H])'''
assert s.count(old) == 1
s = s.replace(old, '''    if prescale:  # sum the k weighted slots, then keep row 0 (the token) of the tile-padded M
        out = ttnn.sum(down, dim=1, keepdim=True)
        if out.shape[-2] != 1:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, 1, H])
        return out
    if down.shape[-2] != 1:  # the compact-A output keeps the tile-padded M of A (32); row 0 is the token
        down = ttnn.slice(down, [0, 0, 0, 0], [1, k, 1, H])''')
open(p, "w").write(s)
print("ok")
