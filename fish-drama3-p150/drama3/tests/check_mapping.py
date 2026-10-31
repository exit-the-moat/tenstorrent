# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU check of the Drama 3 -> qwen36 weight mapping for a few layers (no device).

Builds qwen36's own Qwen36ModelArgs on the shim checkpoint (make_shim_ckpt.py), loads layers
through drama3's per-layer loader (qwen36 dequant_fp8_block + remap_qwen36_state_dict), slices the
per-module substates exactly as tt/layer.py does and checks every key the qwen36 loaders read,
with its shape. The routed experts go through qwen36's own _stack_per_expert and the as_tensor
preprocess transforms of tt/moe/weights.py.

    HF_MODEL=/opt/ep0/drama3_shim python models/demos/blackhole/drama3/tests/check_mapping.py
"""

import os
import sys

import torch

from models.demos.blackhole.drama3.tt.weight_mapping import (
    backbone_keys,
    dequant_fp8,
    load_drama3_state_dict,
    read_tensors,
    weight_map,
)
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.moe import MoEConfig
from models.demos.blackhole.qwen36.tt.moe.weights import _stack_per_expert
from models.demos.blackhole.qwen36.utils.substate import substate

LAYERS = [int(x) for x in os.environ.get("CHECK_LAYERS", "0,3,38,39").split(",")]
failures = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def expect(sd, shapes, where):
    """sd must hold exactly these keys, with these shapes."""
    for k, shp in shapes.items():
        got = tuple(sd[k].shape) if k in sd else None
        check(got == tuple(shp), f"{where}.{k}: {got} == {tuple(shp)}")
    extra = sorted(set(sd) - set(shapes))
    check(not extra, f"{where}: no unconsumed keys {extra[:6]}")


def main():
    args = Qwen36ModelArgs(mesh_device=None, max_batch_size=1, max_seq_len=4096)
    ckpt = args.CKPT_DIR
    print(f"CKPT_DIR {ckpt}  model_name {args.model_name}")
    H, E, I = args.dim, args.moe_num_experts, args.moe_intermediate_size
    check(args.n_layers == 40 and H == 2048, f"n_layers {args.n_layers}, dim {H}")
    check(args.vocab_size == 252288, f"vocab_size {args.vocab_size} (padded {args.padded_vocab_size})")
    check((args.n_heads, args.n_kv_heads, args.head_dim) == (16, 2, 256), "heads 16 / kv 2 / head_dim 256")
    check((E, args.moe_top_k, I) == (256, 8, 512), f"experts {E} top{args.moe_top_k} I={I}")
    check(args.moe_shared_intermediate_size == 512, f"shared I {args.moe_shared_intermediate_size}")
    check(args.moe_norm_topk_prob is True, f"norm_topk_prob {args.moe_norm_topk_prob}")
    check(
        (args.partial_rotary_factor, args.rope_head_dim, args.rope_theta) == (0.25, 64, 10_000_000),
        f"rope {args.partial_rotary_factor} {args.rope_head_dim} {args.rope_theta}",
    )
    full = [i for i in range(args.n_layers) if args.is_full_attention_layer(i)]
    check(full == list(range(3, 40, 4)), f"full attention layers {full}")
    check(all(args.is_moe_layer(i) for i in range(args.n_layers)), "every layer is MoE")
    check(args.norm_eps == 1e-6, f"norm_eps {args.norm_eps}")
    print("MoEConfig", MoEConfig.from_args(args))
    print("GDNConfig", GDNConfig.from_args(args))
    print("AttentionConfig", AttentionConfig.from_args(args))

    wmap = weight_map(ckpt)
    audio = [k for k in wmap if not (k.startswith("model.language_model.") or k == "lm_head.weight")]
    check(all(k.startswith(("audio_decoder.", "audio_decoder_mtp.", "audio_decoder_mtp_shared.")) for k in audio),
          f"{len(audio)} non-backbone keys, all audio_decoder*")

    sd = load_drama3_state_dict(ckpt, LAYERS, embed=True, norm=True, lm_head=True, wmap=wmap)
    expect(
        {k: v for k, v in sd.items() if not k.startswith("layers.")},
        {"tok_embeddings.weight": (252288, H), "norm.weight": (H,), "output.weight": (252288, H)},
        "top",
    )
    qk, v_dim, nv = args.linear_q_dim, args.linear_v_dim, args.linear_num_value_heads
    for L in LAYERS:
        p = f"layers.{L}"
        layer_sd = substate(sd, p)
        expect(substate(layer_sd, "input_layernorm"), {"weight": (H,)}, f"{p}.input_layernorm")
        expect(substate(layer_sd, "post_attention_layernorm"), {"weight": (H,)}, f"{p}.post_attention_layernorm")
        if args.is_full_attention_layer(L):
            expect(
                substate(sd, f"{p}.self_attn"),
                {
                    "q_proj.weight": (args.n_heads * args.head_dim * 2, H),  # query + output gate
                    "k_proj.weight": (args.n_kv_heads * args.head_dim, H),
                    "v_proj.weight": (args.n_kv_heads * args.head_dim, H),
                    "o_proj.weight": (H, args.n_heads * args.head_dim),
                    "q_norm.weight": (args.head_dim,),
                    "k_norm.weight": (args.head_dim,),
                },
                f"{p}.self_attn",
            )
        else:
            k = args.linear_conv_kernel_dim
            expect(
                substate(sd, f"{p}.linear_attn"),
                {
                    "qkv_proj.weight": (2 * qk + v_dim, H),
                    "q_conv.weight": (qk, 1, k),
                    "k_conv.weight": (qk, 1, k),
                    "v_conv.weight": (v_dim, 1, k),
                    "in_proj_a.weight": (nv, H),
                    "in_proj_b.weight": (nv, H),
                    "in_proj_z.weight": (v_dim, H),
                    "out_proj.weight": (H, v_dim),
                    "A_log": (nv,),
                    "dt_bias": (nv,),
                    "norm.weight": (args.linear_value_head_dim,),
                },
                f"{p}.linear_attn",
            )
        mlp = substate(sd, f"{p}.mlp")
        expect(substate(mlp, "gate"), {"weight": (E, H)}, f"{p}.mlp.gate")
        expect(
            substate(mlp, "shared_expert"),
            {"gate_proj.weight": (512, H), "up_proj.weight": (512, H), "down_proj.weight": (H, 512)},
            f"{p}.mlp.shared_expert",
        )
        check(tuple(mlp["shared_expert_gate.weight"].shape) == (1, H), f"{p}.mlp.shared_expert_gate (1, {H})")
        experts = substate(mlp, "experts")
        check(len(experts) == 3 * E, f"{p}.mlp.experts: {len(experts)} per-expert tensors, no scales left")
        check(all(t.dtype == torch.bfloat16 for t in experts.values()), f"{p}.mlp.experts dequantized to bf16")
        fused = _stack_per_expert(experts, I)  # qwen36's own stacking
        check(tuple(fused["gate_up_proj"].shape) == (E, 2 * I, H), f"{p} gate_up_proj {tuple(fused['gate_up_proj'].shape)}")
        check(tuple(fused["down_proj"].shape) == (E, H, I), f"{p} down_proj {tuple(fused['down_proj'].shape)}")
        # tt/moe/weights.py preprocess transforms (cache-miss path)
        gate = fused["gate_up_proj"][:, :I, :].transpose(-2, -1).unsqueeze(0)
        down = fused["down_proj"].transpose(-2, -1).unsqueeze(0)
        check(tuple(gate.shape) == (1, E, H, I) and tuple(down.shape) == (1, E, I, H), f"{p} device layouts")
        check(torch.equal(gate[0, 5], experts["5.gate_proj.weight"].T), f"{p} expert 5 gate rows first in gate_up")

    # qwen36's bf16 dequant vs an fp32 dequant of the same FP8 block (layer 0 expert 0 gate)
    key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    raw = read_tensors(ckpt, wmap, [key, key + "_scale_inv"])
    w32 = dequant_fp8(raw, torch.float32)[key]
    w16 = dequant_fp8(raw)[key].float()
    rel = ((w16 - w32).abs().max() / w32.abs().max()).item()
    check(rel < 4e-3, f"bf16 dequant vs fp32 dequant: max rel err {rel:.2e}")
    print(f"scale_inv {tuple(raw[key + '_scale_inv'].shape)} {raw[key + '_scale_inv'].dtype}, weight {raw[key].dtype}")

    print(f"\n{len(failures)} failures")
    print("DONE" if not failures else "FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
