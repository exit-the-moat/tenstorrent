"""Day 4b: lean GDN recurrent decode step (opt-in QWEN36_GDN_LEAN_DECODE=1): the same ops on the same values,
kept in [B, H, 1, X] so the 3D <-> 4D reshapes of the original drop out."""
p = "/opt/ep0/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py"
s = open(p).read()
old = '''def recurrent_gated_delta_rule_decode_inplace_ttnn(
    q,
    k,
    v,
    beta,
    g,
    state_buffer,
    scale=None,
    device=None,
    high_precision=False,
):
    """Decode with in-place state copy to pre-allocated buffer (trace-safe addresses)."""
'''
assert s.count(old) == 1
s = s.replace(old, '''def recurrent_gated_delta_rule_decode_lean_ttnn(q, k, v, beta, g, state_buffer, scale=None, device=None):
    """recurrent_gated_delta_rule_decode_ttnn + the in-place state copy, with every per-head row kept as
    [B, H, 1, X]: q/k/v/beta/g reshaped once, delta and the outer-product row used as they come, k as a column by
    one transpose. Same values op for op (only layouts differ). q, k: [B, 1, H, K]; v: [B, 1, H, V]."""
    B, H, K, V = q.shape[0], q.shape[2], q.shape[3], v.shape[3]
    L1 = ttnn.L1_MEMORY_CONFIG
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)
    q = ttnn.multiply(q, scale if scale is not None else K**-0.5, memory_config=L1)
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=L1)
    k_row = ttnn.reshape(k, [B, H, 1, K], memory_config=L1)
    v_row = ttnn.reshape(v, [B, H, 1, V], memory_config=L1)
    g4 = ttnn.reshape(g, [B, H, 1, 1], memory_config=L1)
    b4 = ttnn.reshape(beta, [B, H, 1, 1], memory_config=L1)
    h = ttnn.to_memory_config(state_buffer, L1)
    read_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    write_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    read_cfg = None
    if device is not None:
        try:
            read_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass
    h = ttnn.multiply(h, g4, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=L1)
    v_read = ttnn.matmul(k_row, h, memory_config=L1, program_config=read_cfg, compute_kernel_config=read_ckc)
    delta = ttnn.subtract(v_row, v_read, memory_config=L1)
    k_col = ttnn.transpose(k_row, -2, -1, memory_config=L1)
    outer = ttnn.matmul(k_col, delta, memory_config=L1, compute_kernel_config=write_ckc)
    outer = ttnn.multiply(outer, b4, memory_config=L1)
    h = ttnn.add(h, outer, memory_config=L1)
    o = ttnn.matmul(q_row, h, memory_config=L1, program_config=read_cfg, compute_kernel_config=read_ckc)
    o = ttnn.reshape(o, [B, 1, H, V], memory_config=L1)
    ttnn.copy(h, state_buffer)
    ttnn.deallocate(h)
    return o, state_buffer


def recurrent_gated_delta_rule_decode_inplace_ttnn(
    q,
    k,
    v,
    beta,
    g,
    state_buffer,
    scale=None,
    device=None,
    high_precision=False,
):
    """Decode with in-place state copy to pre-allocated buffer (trace-safe addresses)."""
    import os

    if not high_precision and os.environ.get("QWEN36_GDN_LEAN_DECODE", "0") == "1":
        return recurrent_gated_delta_rule_decode_lean_ttnn(q, k, v, beta, g, state_buffer, scale=scale, device=device)
''')
open(p, "w").write(s)
print("ok")
