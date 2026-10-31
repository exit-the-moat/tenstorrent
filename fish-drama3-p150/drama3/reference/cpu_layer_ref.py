# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-by-layer FP32 CPU reference of the Drama 3 backbone for one fixed prompt.

Only one decoder layer's weights are in RAM at a time (the full backbone is ~70 GB in BF16).
Each layer is a transformers ``Qwen3_5MoeDecoderLayer`` built on the meta device and filled with
the checkpoint weights (FP8 experts dequantized in FP32), run once as a fresh prefill.

Usage (inside /opt/ep0/run.sh):
    python -m models.demos.blackhole.drama3.reference.cpu_layer_ref --out /opt/ep0/golden
    python -m models.demos.blackhole.drama3.reference.cpu_layer_ref --selfcheck 4
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from models.demos.blackhole.drama3.tt.weight_mapping import (
    BACKBONE_PREFIX,
    backbone_keys,
    dequant_fp8,
    read_tensors,
    weight_map,
)

MODEL = os.environ.get("DRAMA3_MODEL", "/models/drama-3")
LINE = "So, did we exit the moat? Not yet. But the climb is shorter than I thought."

# reference-bracket-v1 without voice references (sglang_omni/models/fishaudio_drama3/prompt.py):
# each literal is tokenized on its own, because tokenizing the joined string changes ids at the seams.
TTS_INSTRUCTION = "convert the provided text to speech"
PROMPT_SEGMENTS = [
    "<|im_start|>system\n",
    TTS_INSTRUCTION,
    "<|im_end|>\n",
    "<|im_start|>user\n",
    f"<|speaker:0|> {LINE}",
    "<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
]
# Text keys of config.json the transformers config has no field for (see sglang text_config.py).
FISH_ONLY_TEXT_KEYS = (
    "model_type",
    "mlp_layer_types",
    "rotary_interleaved",
    "use_qk_norm",
    "layernorm_zero_centered_gamma",
    "shared_expert_gate",
)


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def vm_hwm_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 2**20
    return -1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(model_path):
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

    full = json.load(open(Path(model_path) / "config.json"))
    text = {k: v for k, v in full["text_config"].items() if k not in FISH_ONLY_TEXT_KEYS}
    # The backbone hard-codes these; refuse a checkpoint that says otherwise.
    tc = full["text_config"]
    assert tc["layernorm_zero_centered_gamma"] and tc["use_qk_norm"] and tc["attn_output_gate"]
    assert not tc["rotary_interleaved"] and tc["shared_expert_gate"]
    cfg = Qwen3_5MoeTextConfig(**text)
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
    return full, cfg


def prompt_ids(model_path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
    ids = []
    for seg in PROMPT_SEGMENTS:
        ids += tok.encode(seg, add_special_tokens=False).ids
    return tok, ids


def hf_layer_state(sd, layer, num_experts):
    """Checkpoint keys of one layer -> Qwen3_5MoeDecoderLayer keys (experts stacked into the fused
    [E, 2I, H] gate_up (gate rows first) / [E, H, I] down layout transformers uses)."""
    p = f"{BACKBONE_PREFIX}layers.{layer}."
    out = {k[len(p) :]: v for k, v in sd.items() if k.startswith(p) and ".mlp.experts." not in k}
    e = f"{p}mlp.experts."
    out["mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat([sd[f"{e}{i}.gate_proj.weight"], sd[f"{e}{i}.up_proj.weight"]], 0) for i in range(num_experts)]
    )
    out["mlp.experts.down_proj"] = torch.stack([sd[f"{e}{i}.down_proj.weight"] for i in range(num_experts)])
    return out


# --quant qwen36: round the weights exactly as the qwen36 single-device modules store them on the card
# (ttnn's own host BFP4/BFP8 conversion of the transposed [in, out] tile layout), keeping FP32 math.
# Separates the weight-format error from the device's math/activation error.
QUANT = None
QWEN36_FORMATS = {
    "linear_attn.in_proj_qkv.weight": "bfloat8_b",
    "linear_attn.in_proj_z.weight": "bfloat8_b",
    "linear_attn.in_proj_a.weight": "bfloat8_b",
    "linear_attn.in_proj_b.weight": "bfloat8_b",
    "linear_attn.out_proj.weight": "bfloat8_b",
    "self_attn.q_proj.weight": "bfloat8_b",
    "self_attn.k_proj.weight": "bfloat8_b",
    "self_attn.v_proj.weight": "bfloat8_b",
    "self_attn.o_proj.weight": "bfloat8_b",
    "mlp.shared_expert.gate_proj.weight": "bfloat4_b",
    "mlp.shared_expert.up_proj.weight": "bfloat4_b",
    "mlp.shared_expert.down_proj.weight": "bfloat8_b",
}


def device_round(w_out_in, fmt):
    """[..., out, in] -> the same values after the device's [..., in, out] BFP tiling and back."""
    import ttnn

    t = ttnn.from_torch(w_out_in.transpose(-2, -1).to(torch.bfloat16), dtype=getattr(ttnn, fmt), layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().transpose(-2, -1).contiguous()


def quantize_like_qwen36(state, cfg, expert_gate_up="bfloat4_b", expert_down="bfloat8_b"):
    for k, fmt in QWEN36_FORMATS.items():
        if k in state:
            state[k] = device_round(state[k], fmt)
    i = cfg.moe_intermediate_size
    gu = state["mlp.experts.gate_up_proj"]  # [E, 2I, H]; gate and up are separate device tensors
    state["mlp.experts.gate_up_proj"] = torch.cat(
        [device_round(gu[:, :i], expert_gate_up), device_round(gu[:, i:], expert_gate_up)], 1
    )
    state["mlp.experts.down_proj"] = device_round(state["mlp.experts.down_proj"], expert_down)
    return state


def build_layer(cfg, layer, wmap):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    raw = read_tensors(MODEL, wmap, backbone_keys(wmap, [layer]))
    sd = dequant_fp8(raw, dtype=torch.float32)
    del raw
    state = {k: v.float() for k, v in hf_layer_state(sd, layer, cfg.num_experts).items()}
    del sd
    if QUANT is not None:
        state = quantize_like_qwen36(state, cfg, *QUANT)
    with torch.device("meta"):
        mod = Qwen3_5MoeDecoderLayer(cfg, layer)
    mod.load_state_dict(state, strict=True, assign=True)
    return mod.eval()


def causal_mask(seq_len):
    m = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    return torch.triu(m, diagonal=1)[None, None]


def rotary(cfg, seq_len):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    pos = torch.arange(seq_len)[None, None, :].expand(3, 1, -1)  # text-only: T = H = W positions
    return rope(torch.zeros(1, dtype=torch.float32), pos)


def run_layers(cfg, wmap, x, layers, on_layer):
    seq = x.shape[1]
    pe = rotary(cfg, seq)
    mask = causal_mask(seq)
    for i in layers:
        t0 = time.time()
        mod = build_layer(cfg, i, wmap)
        t_load = time.time() - t0
        routed = {}
        hook = mod.mlp.gate.register_forward_hook(lambda m, a, o: routed.update(idx=o[2], w=o[1]))
        with torch.no_grad():
            kind = cfg.layer_types[i]
            x = mod(x, position_embeddings=pe, attention_mask=None if kind == "linear_attention" else mask)
        hook.remove()
        del mod
        on_layer(i, x, routed, t_load, time.time() - t0 - t_load)
    return x


def selfcheck(n_layers):
    """The layer loop against transformers' own Qwen3_5MoeTextModel for the first n layers."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    full, cfg = load_config(MODEL)
    wmap = weight_map(MODEL)
    _, ids = prompt_ids(MODEL)
    emb = read_tensors(MODEL, wmap, backbone_keys(wmap, embed=True, norm=True))
    x0 = emb[f"{BACKBONE_PREFIX}embed_tokens.weight"][torch.tensor(ids)].float()[None]

    outs = []
    run_layers(cfg, wmap, x0, range(n_layers), lambda i, x, r, a, b: outs.append(x))

    small = load_config(MODEL)[1]
    small.num_hidden_layers = n_layers
    small.layer_types = cfg.layer_types[:n_layers]
    with torch.device("meta"):
        model = Qwen3_5MoeTextModel(small)
    state = {
        "embed_tokens.weight": emb[f"{BACKBONE_PREFIX}embed_tokens.weight"].float(),
        "norm.weight": emb[f"{BACKBONE_PREFIX}norm.weight"].float(),
    }
    for i in range(n_layers):
        raw = read_tensors(MODEL, wmap, backbone_keys(wmap, [i]))
        for k, v in hf_layer_state(dequant_fp8(raw, torch.float32), i, cfg.num_experts).items():
            state[f"layers.{i}.{k}"] = v.float()
        del raw
    model.load_state_dict(state, strict=False, assign=True)
    model.rotary_emb = type(model.rotary_emb)(small)  # buffers were on meta
    with torch.no_grad():
        res = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    hs = res.hidden_states
    log(f"HF returned {len(hs)} hidden states")
    norm_w = state["norm.weight"]

    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + norm_w)

    for i, o in enumerate(outs):
        # transformers' last hidden_states entry is already final-normed
        ref, o = (hs[i + 1], o) if i + 1 < n_layers else (hs[i + 1], norm(o))
        d = (ref - o).abs().max().item()
        log(f"layer {i} ({cfg.layer_types[i]}): max|HF - loop| = {d:.3e}  (max|x| {o.abs().max().item():.2f})")
    d = (res.last_hidden_state - norm(outs[-1])).abs().max().item()
    log(f"final norm: max|HF - loop| = {d:.3e}")
    log(f"peak RSS {vm_hwm_gb():.1f} GB")


def main_rows(rows_path, out, threads):
    """--rows: the reference for a frame-row prompt (voice reference as semantic rows, optionally followed by
    teacher-forced generated frames), embedded with drama3.host.prompt.FrameEmbedder (Fish's embed_backbone).
    Saves every layer's residual as fp16 (the PCC check needs all rows; fp32 would be ~240 MB), the final norm
    and the candidate logits (eos + semantic band) of every row."""
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    full, cfg = load_config(MODEL)
    wmap = weight_map(MODEL)
    rows = torch.load(rows_path)
    keys = [f"{BACKBONE_PREFIX}embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    emb = FrameEmbedder(t[keys[0]], t[keys[1]], full["semantic_start_token_id"], full["semantic_end_token_id"])
    x = emb(rows)[None]
    del t, emb
    n_sem = int(((rows[:, 0] >= full["semantic_start_token_id"]) & (rows[:, 0] <= full["semantic_end_token_id"])).sum())
    log(f"rows {tuple(rows.shape)} ({n_sem} semantic) -> embeddings {tuple(x.shape)}")
    files = {}

    def save(name, obj):
        path = out / name
        torch.save(obj, path)
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    save("embedding.pt", {"hidden": x[0].clone()})

    def on_layer(i, h, routed, t_load, t_run):
        save(f"layer_{i:02d}.pt", {"hidden": h[0].to(torch.float16)})
        log(f"layer {i:2d} {cfg.layer_types[i]:16s} load {t_load:5.1f}s run {t_run:5.2f}s peak RSS {vm_hwm_gb():.1f} GB")

    x = run_layers(cfg, wmap, x, range(cfg.num_hidden_layers), on_layer)
    rest = read_tensors(MODEL, wmap, backbone_keys(wmap, norm=True, lm_head=True))
    xf = x[0]
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + rest[f"{BACKBONE_PREFIX}norm.weight"].float())
    save("final_norm.pt", {"hidden": normed.clone()})
    eos, sem0, sem1 = full["eos_token_id"], full["semantic_start_token_id"], full["semantic_end_token_id"]
    cand = torch.tensor(sorted({eos, *range(sem0, sem1 + 1)}))
    cl = normed @ rest["lm_head.weight"][cand].float().T
    save("cand_logits.pt", {"logits": cl, "candidates": cand})
    top5 = [int(cand[j]) for j in torch.topk(cl[-1], 5).indices]
    json.dump(
        dict(
            what="Drama 3 backbone FP32 CPU reference for a frame-row prompt (mixed embeddings)",
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            rows=str(rows_path),
            n_rows=rows.shape[0],
            n_semantic_rows=n_sem,
            row_ids=rows[:, 0].tolist(),
            layer_dtype="float16 (residual stream after each layer)",
            top5_last_candidates=top5,
            peak_rss_gb=round(vm_hwm_gb(), 1),
            files=files,
        ),
        open(out / "manifest.json", "w"),
        indent=1,
    )
    log(f"top5 last row: {top5}; peak RSS {vm_hwm_gb():.1f} GB")
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None, help="frame-row prompt (.pt [N, 11]) -> main_rows")
    ap.add_argument("--out", default="/opt/ep0/golden")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (debug)")
    ap.add_argument("--selfcheck", type=int, default=0, help="compare the loop with HF for the first N layers")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--quant",
        default=None,
        help="'qwen36' = experts gate/up BFP4 + down BFP8 (the device default); 'bfp8' = all experts BFP8; "
        "other weights as qwen36 stores them. FP32 math either way.",
    )
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global QUANT
    if a.quant:
        assert a.out != "/opt/ep0/golden", "a quantized run never overwrites the golden reference"
        QUANT = {"qwen36": ("bfloat4_b", "bfloat8_b"), "bfp8": ("bfloat8_b", "bfloat8_b")}[a.quant]
    if a.rows:
        assert a.out != "/opt/ep0/golden", "a --rows run never overwrites the Day 1 golden reference"
        return main_rows(a.rows, a.out, a.threads)
    if a.selfcheck:
        selfcheck(a.selfcheck)
        print("DONE", flush=True)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full, cfg = load_config(MODEL)
    wmap = weight_map(MODEL)
    tok, ids = prompt_ids(MODEL)
    n_layers = a.layers or cfg.num_hidden_layers
    log(f"prompt: {len(ids)} tokens: {ids}")
    files = {}

    def save(name, obj):
        path = out / name
        torch.save(obj, path)
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    emb = read_tensors(MODEL, wmap, backbone_keys(wmap, embed=True))[f"{BACKBONE_PREFIX}embed_tokens.weight"]
    x = emb[torch.tensor(ids)].float()[None]  # text rows only: embed_backbone is the plain embedding
    del emb
    save("embedding.pt", {"hidden": x[0].clone()})
    log(f"embedding {tuple(x.shape)} peak RSS {vm_hwm_gb():.1f} GB")

    layer_log = []

    def on_layer(i, h, routed, t_load, t_run):
        save(
            f"layer_{i:02d}.pt",
            {
                "hidden": h[0].clone(),
                "topk_experts": routed["idx"].to(torch.int16),
                "topk_weights": routed["w"].float(),
            },
        )
        layer_log.append(
            {
                "layer": i,
                "type": cfg.layer_types[i],
                "load_s": round(t_load, 1),
                "run_s": round(t_run, 2),
                "absmax": h.abs().max().item(),
                "rms": h.pow(2).mean().sqrt().item(),
            }
        )
        log(
            f"layer {i:2d} {cfg.layer_types[i]:16s} load {t_load:5.1f}s run {t_run:5.2f}s "
            f"rms {layer_log[-1]['rms']:.3f} absmax {layer_log[-1]['absmax']:.1f} peak RSS {vm_hwm_gb():.1f} GB"
        )

    x = run_layers(cfg, wmap, x, range(n_layers), on_layer)

    rest = read_tensors(MODEL, wmap, backbone_keys(wmap, norm=True, lm_head=True))
    norm_w = rest[f"{BACKBONE_PREFIX}norm.weight"].float()
    xf = x[0]
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + norm_w)
    save("final_norm.pt", {"hidden": normed.clone()})
    w = rest["lm_head.weight"]
    del rest
    logits = torch.cat([normed @ w[s : s + 32768].float().T for s in range(0, w.shape[0], 32768)], dim=-1)
    del w
    last = logits[-1].clone()
    save("logits_last.pt", {"logits": last})

    sem0, sem1, eos = full["semantic_start_token_id"], full["semantic_end_token_id"], full["eos_token_id"]
    cand = torch.tensor(sorted({eos, *range(sem0, sem1 + 1)}))

    def top(v, k=5):
        p = torch.softmax(v, -1)
        vals, idx = torch.topk(v, k)
        return [
            {"id": int(t), "token": tok.id_to_token(int(t)), "logit": round(float(l), 4), "prob": round(float(p[t]), 5)}
            for l, t in zip(vals, idx)
        ]

    cand_logits = last[cand]
    cand_top = top(cand_logits)
    for c in cand_top:  # map candidate-index back to vocab ids
        c["id"] = int(cand[c["id"]])
        c["token"] = tok.id_to_token(c["id"])
    manifest = {
        "what": "Drama 3 backbone FP32 CPU reference, layer by layer, fresh prefill of one prompt",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": "fishaudio/drama-3 (/opt/llm/hf/drama-3)",
        # the worktree's gitdir is not mounted in the container: run.sh callers pass EP0_COMMIT
        "tt_metal_commit": os.environ.get("EP0_COMMIT")
        or subprocess.run(["git", "-C", "/opt/ep0/tt-metal", "rev-parse", "HEAD"], capture_output=True, text=True)
        .stdout.strip(),
        "reference_impl": "transformers %s Qwen3_5MoeDecoderLayer (eager attention, torch GDN), FP32 math; "
        "experts FP8 e4m3 x weight_scale_inv dequantized in FP32; other weights BF16 -> FP32"
        % __import__("transformers").__version__,
        "prompt_template": "reference-bracket-v1, no voice reference, speaker 0, segments tokenized separately",
        "prompt_segments": PROMPT_SEGMENTS,
        "prompt_ids": ids,
        "prompt_tokens": [tok.id_to_token(i) for i in ids],
        "seq_len": len(ids),
        "positions": "0..seq_len-1 (text-only M-RoPE: T=H=W)",
        "dtype": "float32",
        "shapes": {
            "embedding.pt:hidden": [len(ids), cfg.hidden_size],
            "layer_NN.pt:hidden": [len(ids), cfg.hidden_size],
            "layer_NN.pt:topk_experts": [len(ids), cfg.num_experts_per_tok],
            "final_norm.pt:hidden": [len(ids), cfg.hidden_size],
            "logits_last.pt:logits": [cfg.vocab_size],
        },
        "layer_output": "residual stream after layer N (input to layer N+1), before the final norm",
        "num_layers_run": n_layers,
        "top5_last_full_vocab": top(last),
        "top5_last_semantic_candidates": cand_top,
        "semantic_candidates": f"eos {eos} + semantic {sem0}..{sem1} (what Fish's lm_head keeps)",
        "argmax_per_position": [int(t) for t in logits.argmax(-1)],
        "layers": layer_log,
        "peak_rss_gb": round(vm_hwm_gb(), 1),
        "files": files,
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    log(f"top5 full vocab: {manifest['top5_last_full_vocab']}")
    log(f"top5 semantic: {cand_top}")
    log(f"peak RSS {vm_hwm_gb():.1f} GB")
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise
