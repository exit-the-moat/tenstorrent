"""ep3 harness levers applied to a built tt_dit model (no change to tt_dit defaults). Each lever is opt-in from gen.py / op_prof.py."""
import ttnn
import models.tt_dit.layers.linear as L

FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi3": ttnn.MathFidelity.HiFi3, "hifi4": ttnn.MathFidelity.HiFi4}


def iter_modules(m, seen=None):
    seen = seen if seen is not None else set()
    if id(m) in seen:
        return
    seen.add(id(m))
    yield m
    for _, c in m.named_children():
        yield from iter_modules(c, seen)


def set_linear_fidelity(model, fid, arch, skip=()):
    """Math fidelity of every Linear-style layer's matmul (fp32 dest + packer L1 acc kept as in tt_dit)."""
    cc = ttnn.init_device_compute_kernel_config(arch, math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    n = 0
    for m in iter_modules(model):
        if isinstance(m, (L.Linear, L.ColParallelLinear, L.RowParallelLinear)) and hasattr(m, "compute_config"):
            m.compute_config = cc
            n += 1
    return n


def set_sdpa(model, q_chunk, k_chunk, fid=None, grid=None, exp_approx=False):
    """Joint-attention chunk sizes (and optionally fidelity, core grid, exp approximation) on every attention block."""
    n = 0
    for m in iter_modules(model):
        if hasattr(m, "sdpa_program_config") and hasattr(m, "sdpa_worker_grid"):
            m.sdpa_program_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid or m.sdpa_worker_grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=exp_approx)
            if fid:
                m.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False)
            n += 1
    return n


def set_norm_ccl(model, ccl_manager):
    """Give the distributed norm layers their own CCL manager (e.g. fewer links than the all-gathers use)."""
    n = 0
    for m in iter_modules(model):
        if type(m).__name__ in ("DistributedRMSNorm", "DistributedLayerNorm") and hasattr(m, "ccl_manager"):
            m.ccl_manager = ccl_manager
            n += 1
    return n


def set_fuse(model, flags):
    """opt-in fused block forward on every QwenImage block; flags e.g. {"addcmul","attn_out","qkv","ff1","ffrs"}."""
    n = 0
    for m in iter_modules(model):
        if type(m).__name__ == "QwenImageTransformerBlock":
            m.ep3_fuse = set(flags)
            m._ep3_idx = n
            n += 1
    return n


def upcast_weights(model, names=("to_out", "to_add_out")):
    """cast the named Linear weights from bf8 to bf16 on the card after load (bf8 values are exact in bf16, so the
    matmul sees the same weights). Needed by the fused gated-add matmul epilogue (residual dtype must equal weight dtype)."""
    n = 0
    for m in iter_modules(model):
        for nm in names:
            lin = getattr(m, nm, None)
            if lin is not None and hasattr(lin, "weight") and lin.weight.data.get_dtype() == ttnn.bfloat8_b:
                old = lin.weight.data
                lin.weight._data = ttnn.typecast(old, ttnn.bfloat16)
                ttnn.deallocate(old)
                n += 1
    return n


GELU = {"exact": (ttnn.UnaryOpType.GELU, False), "tanh": ttnn.UnaryOpType.GELU_TANH, "fast": (ttnn.UnaryOpType.GELU, True)}


def set_gelu(model, variant):
    """fused activation of the feed-forward ff1 layers. Qwen-Image (diffusers) uses gelu-approximate = tanh;
    tt_dit fuses the exact erf GELU. 'tanh' matches the reference model exactly."""
    n = 0
    for m in iter_modules(model):
        if type(m).__name__ == "ParallelFeedForward" or type(m).__name__ == "FeedForward":
            m.ff1.fused_activation_fn = GELU[variant]
            n += 1
    return n


def set_norm_fidelity(model, fid, fp32_acc=True):
    """math fidelity of the distributed RMS norms (adaLN norms and the qknr fused q/k norm+rope), default HiFi4/fp32 acc.
    Numerics change when not HiFi4. Returns the number of norm modules changed."""
    n = 0
    cc = None
    for m in iter_modules(model):
        if type(m).__name__ == "DistributedRMSNorm":
            if cc is None:
                cc = ttnn.init_device_compute_kernel_config(m.mesh_device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                            math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False)
            m.compute_kernel_config = cc
            n += 1
        if type(m).__name__ == "QwenImageTransformerBlock":
            m._ep3_norm_cc = (fid, fp32_acc)
    return n
