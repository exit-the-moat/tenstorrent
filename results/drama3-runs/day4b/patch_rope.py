"""Day 4b: partial RoPE as one matmul for one-token decode (opt-in via register_partial_rope + full-width cos/sin)."""
R = "/opt/ep0/tt-metal/models/"


def sub(path, old, new, count=1):
    s = open(R + path).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(R + path, "w").write(s.replace(old, new))


sub(
    "experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py",
    '''def apply_rotary_pos_emb_ttnn(q, k, cos, sin):
    """Apply RoPE to query and key tensors using TTNN ops."""
''',
    '''_PARTIAL_ROPE_P = {}  # head_dim -> device [1, 1, D, D] matrix: x @ P = rotate_half(x[:rd]) | 0 (register_partial_rope)


def register_partial_rope(device, head_dim, rotary_dim):
    """Opt-in (single device): with this registered, a call whose cos / sin are FULL width
    ([cos(rd) | 1], [sin(rd) | 0]) applies the partial rotary as x * C + (x @ P) * S, 4 ops per tensor instead
    of the slices / neg / concats (same values: P only moves and negates entries, the 1 / 0 are exact)."""
    import torch

    half = rotary_dim // 2
    p = torch.zeros(head_dim, head_dim)
    for i in range(half):
        p[i + half, i] = -1.0  # out[i] = -x[i + half]
        p[i, i + half] = 1.0  # out[i + half] = x[i]
    _PARTIAL_ROPE_P[head_dim] = (
        ttnn.from_torch(
            p.reshape(1, 1, head_dim, head_dim).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        ttnn.init_device_compute_kernel_config(  # HiFi4: the +-1 products keep every mantissa bit of x
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
    )


def apply_rotary_pos_emb_ttnn(q, k, cos, sin):
    """Apply RoPE to query and key tensors using TTNN ops."""
    if cos.shape[-1] == q.shape[-1] and q.shape[-1] in _PARTIAL_ROPE_P:
        if len(cos.shape) < 4:
            cos = ttnn.unsqueeze(cos, 1)
            sin = ttnn.unsqueeze(sin, 1)
        p, ckc = _PARTIAL_ROPE_P[q.shape[-1]]
        q_embed = ttnn.add(ttnn.multiply(q, cos), ttnn.multiply(ttnn.matmul(q, p, compute_kernel_config=ckc), sin))
        k_embed = ttnn.add(ttnn.multiply(k, cos), ttnn.multiply(ttnn.matmul(k, p, compute_kernel_config=ckc), sin))
        return q_embed, k_embed
''',
)
print("ok")
