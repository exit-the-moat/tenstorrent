p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/decode.py"
s = open(p).read()
old = """    if os.environ.get("QWEN36_MOE_DECODE_INDEXED_DEBUG"):
        print("indexed shapes", tuple(hidden_states.shape), tuple(down_input.shape) if False else "", tuple(down.shape), flush=True)
    next_states = ttnn.reshape(down, (1, k, H))
"""
assert old in s
s = s.replace(old, """    if down.shape[-2] != 1:  # the compact-A output keeps the tile-padded M of A (32); row 0 is the token
        down = ttnn.slice(down, [0, 0, 0, 0], [1, k, 1, H])
    next_states = ttnn.reshape(down, (1, k, H))
""")
s = s.replace("""    )  # [1, k, 1, H]: slot i = expert top_k_indices[i]""", """    )  # [1, k, M, H]: slot i = expert top_k_indices[i]""")
open(p, "w").write(s)
print("ok")
