"""Day 4b: opt-in indexed (gather) MoE decode for one device, batch 1 (QWEN36_MOE_DECODE_INDEXED=1)."""
R = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/"


def sub(path, old, new):
    s = open(R + path).read()
    assert s.count(old) == 1, (path, old[:60])
    open(R + path, "w").write(s.replace(old, new))


# router: expose the normalised top-k
sub(
    "router.py",
    '''    def __call__(self, hidden_states):
        """hidden_states: [1,1,S,H] (replicated full hidden). Returns [1,1,S,E]."""
        expert_scores''',
    '''    def topk(self, hidden_states):
        """hidden_states: [1,1,S,H] -> (top-k weights [1,1,S,k] (sum-normalised if norm_topk_prob), expert ids
        [1,1,S,k] uint16), the dense routing's ingredients without the scatter."""
        expert_scores = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)
        router_probs.deallocate(True)
        if self.norm_topk_prob:
            top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
            top_k_values = ttnn.div(top_k_values, top_k_sum)
            top_k_sum.deallocate(True)
        return top_k_values, top_k_indices

    def __call__(self, hidden_states):
        """hidden_states: [1,1,S,H] (replicated full hidden). Returns [1,1,S,E]."""
        expert_scores''',
)

# decode: the indexed path
sub(
    "decode.py",
    '''def decode_forward(
    hidden_states,''',
    '''def decode_forward_indexed(hidden_states, top_k_values, top_k_indices, weights: ExpertWeights, config, sparsity):
    """Single device, one token: only the top-k experts, via sparse_matmul's indexed/gather mode (compact
    [.., k, M, N] outputs), so neither the matmuls nor the glue touch the other experts' slots (the dense path
    writes, permutes and sums all E = 256 slots). top_k_values [1,1,1,k] (normalised), top_k_indices [1,1,1,k]
    uint16 (router.topk); sparsity: any [1,1,1,E] bf16 ROW_MAJOR tensor (required operand, not read)."""
    k, H = config.top_k, config.hidden_size
    intermediate_size = weights.intermediate_size_per_device
    idx = ttnn.to_layout(top_k_indices, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(
        1, 2 * intermediate_size, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_GATE_UP", "1"))
    )
    down_config = _build_sparse_matmul_config(1, H, in0_block_w=int(os.environ.get("QWEN36_MOE_DECODE_BLOCK_DOWN", "1")))
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
    )  # [1, 1, 1, k, 1, 2I]
    up_gate = ttnn.reshape(up_gate, (1, k, 1, up_gate.shape[-1]))
    down_input = apply_swiglu(up_gate)  # [1, k, 1, I]
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
    )  # [1, k, 1, H]: slot i = expert top_k_indices[i]
    down_input.deallocate(True)
    ttnn.deallocate(idx)
    next_states = ttnn.reshape(down, (1, k, H))
    next_states = ttnn.mul(next_states, ttnn.reshape(top_k_values, (1, k, 1)))
    next_states = ttnn.sum(next_states, dim=1)
    return ttnn.reshape(next_states, (1, 1, 1, H))


def decode_forward(
    hidden_states,''',
)

# experts: flag + the dummy sparsity operand (allocated at build time, i.e. before any trace capture)
sub(
    "experts.py",
    '''from .decode import decode_forward
''',
    '''import os

import torch

import ttnn

from .decode import decode_forward, decode_forward_indexed
''',
)
sub(
    "experts.py",
    '''        self.grouped_prefill = False
        self.prefill_valid_len = None
''',
    '''        self.grouped_prefill = False
        self.prefill_valid_len = None
        # opt-in (single device, batch 1): decode only the top-k experts (QWEN36_MOE_DECODE_INDEXED=1)
        self.indexed_decode = self.num_devices == 1 and os.environ.get("QWEN36_MOE_DECODE_INDEXED", "0") == "1"
        if self.indexed_decode:
            self.decode_sparsity = ttnn.from_torch(
                torch.ones(1, 1, 1, config.num_experts, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )

    def decode_indexed(self, hidden_states, top_k_values, top_k_indices):
        return decode_forward_indexed(hidden_states, top_k_values, top_k_indices, self.weights, self.config, self.decode_sparsity)
''',
)

# moe: use it for single-device one-token decode
sub(
    "moe.py",
    '''        dense_routing = self.router(x4)
        out = self.experts(x4, dense_routing, mode=mode)''',
    '''        if mode == "decode" and single and S == 1 and self.experts.indexed_decode:
            vals, idx = self.router.topk(x4)
            out = self.experts.decode_indexed(x4, vals, idx)
            ttnn.deallocate(vals)
            ttnn.deallocate(idx)
        else:
            dense_routing = self.router(x4)
            out = self.experts(x4, dense_routing, mode=mode)''',
)
print("patched")
