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


GA = "experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py"
edit(
    GA,
    [
        (
            "    chunk_start_idx_tensor=None,  # device tensor [1] int32 — runtime offset for trace-replay (flexible SDPA)\n):\n",
            "    chunk_start_idx_tensor=None,  # device tensor [1] int32 — runtime offset for trace-replay (flexible SDPA)\n"
            "    rows_are_batch=False,  # drama3 batch mode: hidden [1, B, H] = B sequences of one decode token each\n):\n",
        ),
        (
            "    B = hidden_states.shape[0]\n    T = hidden_states.shape[1]\n    scaling = head_dim**-0.5\n",
            "    B = hidden_states.shape[0]\n    T = hidden_states.shape[1]\n"
            "    if rows_are_batch:  # the projections run on the one-tile [1, B, H]; heads / rope / cache per sequence\n"
            "        B, T = T, 1\n"
            "    scaling = head_dim**-0.5\n",
        ),
        (
            "    # Output projection\n    attn_output = ttnn.linear(attn_output, o_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)\n",
            "    # Output projection\n"
            "    if rows_are_batch:  # [B, 1, H*D] -> the one-tile [1, B, H*D]\n"
            "        attn_output = ttnn.reshape(attn_output, [1, B, attn_output.shape[-1]])\n"
            "    attn_output = ttnn.linear(attn_output, o_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)\n",
        ),
    ],
)
edit(
    "demos/blackhole/qwen36/tt/attention/decode.py",
    [
        (
            "    paged_kv_cache_value=None,\n):\n",
            "    paged_kv_cache_value=None,\n    rows_are_batch=False,\n):\n",
        ),
        (
            "        paged_kv_cache_value=paged_kv_cache_value,\n    )\n",
            "        paged_kv_cache_value=paged_kv_cache_value,\n        rows_are_batch=rows_are_batch,\n    )\n",
        ),
    ],
)
edit(
    "demos/blackhole/qwen36/tt/attention/gated_attention.py",
    [
        (
            "        T = x.shape[1]\n        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None\n",
            "        # rows_are_batch (attribute, drama3 batch mode): x [1, B, H] holds B one-token decode rows\n"
            "        rows = getattr(self, \"rows_are_batch\", False)\n"
            "        T = 1 if rows else x.shape[1]\n        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None\n",
        ),
        (
            "                paged_kv_cache_value=self.paged_kv_cache_value,\n            )\n        elif self.use_paged_attention and T > 1 and chunk_page_table is not None:\n",
            "                paged_kv_cache_value=self.paged_kv_cache_value,\n                rows_are_batch=rows,\n            )\n"
            "        elif self.use_paged_attention and T > 1 and chunk_page_table is not None:\n",
        ),
    ],
)
GD = "experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py"
edit(
    GD,
    [
        (
            "    valid_len=None,  # fixed-bucket padding; zeros padded positions in scan\n):\n",
            "    valid_len=None,  # fixed-bucket padding; zeros padded positions in scan\n"
            "    rows_are_batch=False,  # drama3 batch mode: hidden [1, B, H] = B sequences of one decode token each\n):\n",
        ),
        (
            "    B = hidden_states.shape[0]\n    T = hidden_states.shape[1]\n    # valid_len path forces DRAM",
            "    B = hidden_states.shape[0]\n    T = hidden_states.shape[1]\n"
            "    if rows_are_batch:  # projections / conv / gates on the one-tile [1, B, .]; the recurrence per sequence\n"
            "        B, T = T, 1\n"
            "    # valid_len path forces DRAM",
        ),
        (
            "    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim], memory_config=mc)\n    if mc is not None:\n",
            "    o = ttnn.reshape(o, [1, B, num_v_heads * head_v_dim] if rows_are_batch else [B, T, num_v_heads * head_v_dim], memory_config=mc)\n"
            "    if mc is not None:\n",
        ),
    ],
)
edit(
    "demos/blackhole/qwen36/tt/gdn/decode.py",
    [
        (
            "        valid_len=valid_len,\n    )\n",
            "        valid_len=valid_len,\n        rows_are_batch=getattr(gdn, \"rows_are_batch\", False),  # drama3 batch mode (attribute, default off)\n    )\n",
        ),
    ],
)
print("PATCHED")
