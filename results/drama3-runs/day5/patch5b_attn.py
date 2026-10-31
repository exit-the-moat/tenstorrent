import os

os.chdir("/opt/ep0/tt-metal/models")


def edit(p, pairs):
    raw = open(p, "rb").read()
    crlf = b"\r\n" in raw
    s = raw.decode().replace("\r\n", "\n")
    for o, n in pairs:
        assert s.count(o) == 1, (p, o)
        s = s.replace(o, n)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode())


edit(
    "experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py",
    [
        (
            "        q_decode = ttnn.transpose(query_states, 1, 2)  # [B, H_q, 1, D] -> [B, 1, H_q, D] = [1, B, H_q, D] for B=1\n",
            "        q_decode = ttnn.transpose(query_states, 1, 2)  # [B, H_q, 1, D] -> [B, 1, H_q, D] = [1, B, H_q, D] for B=1\n"
            "        if B > 1:  # several decode rows (drama3 batch mode): sdpa decode wants [1, B, H_q, D]\n"
            "            q_decode = ttnn.reshape(q_decode, [1, B, num_attention_heads, head_dim])\n",
        ),
        (
            "        attn_output = ttnn.transpose(attn_output, 1, 2)  # back to [B, H_q, 1, D]\n",
            "        if B > 1:  # [1, B, H_q, D] -> [B, 1, H_q, D]\n"
            "            attn_output = ttnn.reshape(attn_output, [B, 1, num_attention_heads, head_dim])\n"
            "        attn_output = ttnn.transpose(attn_output, 1, 2)  # back to [B, H_q, 1, D]\n",
        ),
    ],
)
edit(
    "demos/blackhole/drama3/tt/decode_norm.py",
    [
        (
            "ENABLED = os.environ.get(\"DRAMA3_SHARDED_DECODE_NORM\", \"1\") == \"1\"\n",
            "ENABLED = os.environ.get(\"DRAMA3_SHARDED_DECODE_NORM\", \"1\") == \"1\"\n"
            "# rows of a one-tile decode input that take the sharded path (batch mode sets it to B; default: one row)\n"
            "ROWS = 1\n",
        ),
        (
            "        if x.shape[-2] == 1 and not in_sharded and not out_sharded:\n",
            "        if x.shape[-2] <= ROWS and not in_sharded and not out_sharded:\n",
        ),
    ],
)
edit(
    "demos/blackhole/drama3/tt/decode_linear.py",
    [
        (
            "_installed = False\n",
            "_installed = False\n"
            "# rows of a one-tile input that take the tuned programs (batch mode sets it to B; default: one row)\n"
            "ROWS = 1\n",
        ),
        (
            "                and args[0].shape[-2] == 1\n",
            "                and (args[0].shape[-2] == 1 or (args[0].shape[-2] <= ROWS and math.prod(args[0].shape[:-2]) == 1))\n",
        ),
    ],
)
print("PATCHED")
