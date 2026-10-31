# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fish Audio Drama 3 (fish_omni) checkpoint -> qwen36 internal state dict, one layer at a time.

The backbone ("text model") of Drama 3 is a Qwen3.5-MoE (Qwen3.6-35B-A3B layout) whose weights keep
their Hugging Face names under ``model.language_model.``. The routed experts are block-FP8
(float8_e4m3fn + ``weight_scale_inv`` per 128x128 block), everything else is BF16. The checkpoint
also holds the audio side (``audio_decoder*``), which is not part of the backbone.

qwen36's own FP8 loader (``load_qwen36_state_dict_fp8``) reads the WHOLE checkpoint and keeps the
GDN weights raw for the TP loaders; on one card we need (a) only a few layers in RAM at a time and
(b) the single-device key scheme of ``remap_qwen36_state_dict`` (fused ``qkv_proj``, split conv).
So this module selects keys per layer, dequantizes with qwen36's ``dequant_fp8_block`` and hands
the result to ``remap_qwen36_state_dict`` unchanged.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

BACKBONE_PREFIX = "model.language_model."
AUDIO_PREFIXES = ("audio_decoder.", "audio_decoder_mtp.", "audio_decoder_mtp_shared.")


def weight_map(model_path) -> Dict[str, str]:
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def backbone_keys(
    wmap: Dict[str, str], layers: Iterable[int] = (), embed=False, norm=False, lm_head=False, experts=True
):
    """Checkpoint keys of the selected backbone parts (scale tensors included). experts=False leaves out
    the routed experts (95 % of a layer) for loads whose expert tensors come from the tensor cache."""
    prefixes = tuple(f"{BACKBONE_PREFIX}layers.{i}." for i in layers)
    keys = [k for k in wmap if prefixes and k.startswith(prefixes) and (experts or ".mlp.experts." not in k)]
    if embed:
        keys.append(f"{BACKBONE_PREFIX}embed_tokens.weight")
    if norm:
        keys.append(f"{BACKBONE_PREFIX}norm.weight")
    if lm_head:
        keys.append("lm_head.weight")
    return keys


def read_tensors(model_path, wmap: Dict[str, str], keys) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    by_file: Dict[str, list] = {}
    for k in keys:
        by_file.setdefault(wmap[k], []).append(k)
    out = {}
    for filename, ks in by_file.items():
        with safe_open(str(Path(model_path) / filename), framework="pt") as sf:
            for k in ks:
                out[k] = sf.get_tensor(k)
    return out


def dequant_fp8(raw: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Dequantize every float8 weight with its ``weight_scale_inv`` (dropping the scales).

    dtype=None uses qwen36's ``dequant_fp8_block`` (BF16 result, the product rounded to BF16 —
    what the device conversion will see). dtype=torch.float32 multiplies in FP32 for the CPU
    reference (FP8 values are exact in both; only the scale product differs).
    """
    out = {}
    for key, t in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if t.dtype != torch.float8_e4m3fn:
            out[key] = t
            continue
        scale = raw[key + "_scale_inv"]
        if dtype is None:
            from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

            out[key] = dequant_fp8_block(t, scale)
        else:
            o, i = t.shape
            w = t.to(dtype).reshape(o // 128, 128, i // 128, 128) * scale.to(dtype)[:, None, :, None]
            out[key] = w.reshape(o, i)
    return out


def load_drama3_state_dict(
    model_path, layers: Iterable[int] = (), embed=False, norm=False, lm_head=False, wmap=None, experts=True
) -> Dict[str, torch.Tensor]:
    """qwen36 single-device state dict (``layers.N.*``, ``tok_embeddings``, ``norm``, ``output``)
    for the selected parts of a Drama 3 checkpoint. Audio-side weights are never read.

    experts=False: no ``mlp.experts.*`` keys; qwen36's load_expert_weights then reads the experts from
    the tensor cache (an empty expert substate passes None to ttnn.as_tensor, a cache hit)."""
    from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

    wmap = wmap or weight_map(model_path)
    raw = read_tensors(model_path, wmap, backbone_keys(wmap, layers, embed, norm, lm_head, experts))
    return remap_qwen36_state_dict(dequant_fp8(raw))
