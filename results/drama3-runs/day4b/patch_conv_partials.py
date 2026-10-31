"""Day 4b: GDN decode causal conv as running partial sums (opt-in: conv state list marked .partial_sums)."""
p = "/opt/ep0/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py"
s = open(p).read()
old = '''def _causal_conv1d_decode_t1_split_inplace(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Split-state T=1 conv; inplace copy for trace-stable addresses."""
    mc = memory_config
'''
assert s.count(old) == 1
s = s.replace(old, '''class ConvPartialSums(list):
    """Decode conv state held as running partial sums Q[0..K-2] instead of the last K-1 inputs:
    Q[j] = sum_{i <= K-2-j} w[i] * x[t-(K-2-j)+i] (+ bias): the parts of the next K-1 outputs already known.
    One step: y = Q[0] + w[K-1] * x;  Q[j] = Q[j+1] + w[K-2-j] * x;  Q[K-2] = w[0] * x (+ bias)."""

    partial_sums = True


def _conv_partials_step(x, q, kernel_size, mc, weight_taps, bias_dev):
    K = kernel_size
    out = ttnn.addcmul(q[0], x, weight_taps[K - 1], value=1.0, memory_config=mc)
    for j in range(K - 2):
        ttnn.addcmul(q[j + 1], x, weight_taps[K - 2 - j], value=1.0, output_tensor=q[j])
    if bias_dev is not None:
        ttnn.addcmul(bias_dev, x, weight_taps[0], value=1.0, output_tensor=q[K - 2])
    else:
        ttnn.multiply(x, weight_taps[0], output_tensor=q[K - 2])
    return ttnn.silu(out, memory_config=mc), q


def conv_partials_from_states(states, weight_taps, bias_dev, q):
    """Fill the partial sums q (ConvPartialSums of K-1 [B,1,D]) from the last K-1 inputs states[0..K-2]
    (oldest first), in place."""
    K = len(states) + 1
    for j in range(K - 1):  # q[j] = sum_i w[i] * states[j + i] over the states that remain
        acc = None
        for i in range(K - 1 - j):
            term = ttnn.multiply(states[j + i], weight_taps[i])
            acc = term if acc is None else ttnn.add(acc, term)
        if bias_dev is not None:
            acc = ttnn.add(acc, bias_dev)
        ttnn.copy(acc, q[j])


def _causal_conv1d_decode_t1_split_inplace(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Split-state T=1 conv; inplace copy for trace-stable addresses."""
    mc = memory_config
    if getattr(conv_state_list, "partial_sums", False):
        return _conv_partials_step(x, conv_state_list, kernel_size, mc, weight_taps, bias_dev)
''')
open(p, "w").write(s)
print("ok")
