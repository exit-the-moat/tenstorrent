"""wan2: CPU-only check of tt_dit's adapter_loader vs the cpu_ref LoRA merge (keys, alphas, scales, fused-QKV layout)."""
import re, torch
from safetensors.torch import load_file
from models.tt_dit.experimental.lora import adapter_loader as AL

L = "/models/lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/"
n_dev, heads, hd, dim = 2, 40, 128, 5120
nl = heads // n_dev

def interleave(ts):  # = WanAttention._interleave_heads, ts in [out,in]
    ts = [t.T.reshape(t.shape[1], n_dev, nl, hd) for t in ts]
    m = torch.cat(ts, dim=2).reshape(ts[0].shape[0], len(ts) * heads * hd)
    return m.T

for f in ["high_noise_model", "low_noise_model"]:
    raw = load_file(L + f + ".safetensors")
    pairs, alphas, sk = AL._collect_pairs(raw)
    mapped, scales, unm = {}, {}, []
    for base, ab in pairs.items():
        d = AL._diffusers_path(base)
        if d is None:
            unm.append(base); continue
        mapped[d] = ab
        r = ab["A"].shape[0]
        m = re.match(r"blocks\.(\d+)\.(attn1|attn2)\.to_([qkv])$", d)
        if m and (m.group(2) == "attn1" or m.group(3) in "kv"):
            side = "self_attn" if m.group(2) == "attn1" else "cross_attn"
            key = f"blocks.{m.group(1)}.{side}.q" if side == "self_attn" else f"blocks.{m.group(1)}.{side}.k"
            scales[d] = alphas.get(key, float(r)) / r
        else:
            scales[d] = alphas.get(AL._lightx2v_for_path(d), float(r)) / r
    vals = sorted(set(round(v, 6) for v in scales.values()))
    a_all = sorted(set(alphas.values()))
    print(f, "pairs", len(pairs), "mapped", len(mapped), "unmapped", len(unm), unm[:3], "alphas", a_all, "eff scales", vals, "direct", sk)
    # per-q/k/v alphas equal? (fused group uses the first one)
    diff_alpha = [k for k in alphas if alphas[k] != a_all[0]]
    print("  alphas differing from first:", len(diff_alpha))
    # fused QKV layout check on block 0 and 39, self + cross
    for b in (0, 39):
        for attn, req in (("attn1", "qkv"), ("attn2", "kv")):
            abs_ = [mapped[f"blocks.{b}.{attn}.to_{x}"] for x in req]
            r = abs_[0]["A"].shape[0]; n = len(req)
            A_f = torch.cat([ab["A"] for ab in abs_], 0)
            Bp = []
            for i, ab in enumerate(abs_):
                p = torch.zeros(dim, n * r); p[:, i * r:(i + 1) * r] = ab["B"]; Bp.append(p)
            B_f = AL._head_interleave_lora_B(Bp, n_dev=n_dev, n_local_heads=nl, head_dim=hd)
            tt_delta = (B_f @ A_f)  # [n*out, in] torch convention
            ref = interleave([ab["B"] @ ab["A"] for ab in abs_])
            print(f"  block {b} {attn} fused delta max|diff| {float((tt_delta - ref).abs().max()):.3e}  |ref| {float(ref.abs().max()):.3e}")
    # delta magnitude per group (Frobenius of scale*up@down), block 0 / 20 / 39
    for b in (0, 20, 39):
        s = []
        for g in ["attn1.to_q", "attn1.to_out.0", "attn2.to_k", "attn2.to_out.0", "ffn.net.0.proj", "ffn.net.2"]:
            ab = mapped[f"blocks.{b}.{g}"]
            dlt = 0.125 * ab["B"].float() @ ab["A"].float()
            s.append(f"{g} {float(dlt.norm()):.2f}/{float(dlt.abs().max()):.3f}")
        print(f"  block {b} |delta| fro/max:", "; ".join(s))
print("KEYCHECK_DONE")
