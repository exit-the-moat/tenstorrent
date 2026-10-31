import os

os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe")


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


BATCHED = '''

_BLOCK_MASKS = {}


def _block_mask(device, B, k):
    """[1, B*k, 32, 1] bf16: slot j (row j // k's j % k-th expert) keeps only row j // k of the M tile."""
    key = (B, k)
    if key not in _BLOCK_MASKS:
        import torch

        m = torch.zeros(1, B * k, 32, 1)
        for j in range(B * k):
            m[0, j, j // k, 0] = 1.0
        _BLOCK_MASKS[key] = ttnn.from_torch(m.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return _BLOCK_MASKS[key]


def decode_forward_indexed_batched(hidden_states, top_k_values, top_k_indices, weights: ExpertWeights, config, sparsity):
    """decode_forward_indexed for B decode rows (single device, B <= 32, QWEN36_MOE_DECODE_INDEXED_BATCH=1): the
    B*k routed ids of all rows go to sparse_matmul's gather mode as one index list, every slot computes the whole
    one-tile M (B rows, the extra rows are free), and a block mask times the routing weights keeps row b's own k
    slots before the slot sum. hidden_states [1,1,B,H]; top_k_values / _indices [1,1,B,k] -> [1,1,B,H]."""
    k, H = config.top_k, config.hidden_size
    B = hidden_states.shape[-2]
    intermediate_size = weights.intermediate_size_per_device
    idx = ttnn.reshape(ttnn.to_layout(top_k_indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, B * k))
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(
        B, 2 * intermediate_size, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_GATE_UP", "1"))
    )
    down_config = _build_sparse_matmul_config(B, H, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_DOWN", "1")))
    up_gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        indices=idx,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=expert_decode_compute_kernel_config(),
        dtype=ttnn.bfloat16,
    )  # [1, 1, 1, B*k, M, 2I]
    up_gate = ttnn.reshape(up_gate, (1, B * k, up_gate.shape[-2], up_gate.shape[-1]))
    down_input = apply_swiglu(up_gate)  # [1, B*k, M, I]
    up_gate.deallocate(True)
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        indices=idx,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        compute_kernel_config=expert_decode_compute_kernel_config(),
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )  # [1, B*k, M, H]
    down_input.deallocate(True)
    ttnn.deallocate(idx)
    w = ttnn.multiply(_block_mask(hidden_states.device(), B, k), ttnn.reshape(top_k_values, (1, B * k, 1, 1)))
    if down.shape[-2] != 32:
        down = ttnn.reshape(down, (1, B * k, down.shape[-2], H), (1, B * k, 32, H))
    out = ttnn.sum(ttnn.multiply(down, w), dim=1, keepdim=True)  # [1, 1, M, H]
    ttnn.deallocate(down)
    ttnn.deallocate(w)
    return ttnn.slice(out, [0, 0, 0, 0], [1, 1, B, H])
'''

edit(
    "decode.py",
    [
        (
            "\n\ndef decode_forward(\n",
            BATCHED + "\n\ndef decode_forward(\n",
        ),
    ],
)
edit(
    "experts.py",
    [
        (
            "from .decode import decode_forward, decode_forward_indexed\n",
            "from .decode import decode_forward, decode_forward_indexed, decode_forward_indexed_batched\n",
        ),
        (
            "    def decode_indexed(self, hidden_states, top_k_values, top_k_indices):\n",
            "    def decode_indexed(self, hidden_states, top_k_values, top_k_indices):\n"
            "        if hidden_states.shape[-2] > 1:  # QWEN36_MOE_DECODE_INDEXED_BATCH=1 (moe.py): B decode rows\n"
            "            return decode_forward_indexed_batched(\n"
            "                hidden_states, top_k_values, top_k_indices, self.weights, self.config, self.decode_sparsity\n"
            "            )\n",
        ),
    ],
)
edit(
    "moe.py",
    [
        (
            '        if mode == "decode" and single and S == 1 and self.experts.indexed_decode:\n',
            '        # QWEN36_MOE_DECODE_INDEXED_BATCH=1 (opt-in): the indexed top-k decode for B > 1 rows too\n'
            '        batched = S > 1 and os.environ.get("QWEN36_MOE_DECODE_INDEXED_BATCH", "0") == "1"\n'
            '        if mode == "decode" and single and (S == 1 or batched) and self.experts.indexed_decode:\n',
        ),
    ],
)
s = open("moe.py").read()
if "\nimport os\n" not in s:
    nl = "\r\n" if "\r\n" in s else "\n"
    s = s.replace("import ttnn", "import os" + nl + nl + "import ttnn", 1) if s.startswith("import ttnn") else s
    if "\nimport os" not in s:
        i = s.index("import ttnn")
        s = s[:i] + "import os" + nl + nl + s[i:]
    open("moe.py", "w", newline="").write(s)
print("PATCHED")
