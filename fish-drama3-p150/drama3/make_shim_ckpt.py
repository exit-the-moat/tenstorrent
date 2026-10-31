# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Make a qwen3_5_moe view of the Drama 3 checkpoint for qwen36's Qwen36ModelArgs.

transformers has no ``fish_omni`` model type, so ModelArgs' AutoConfig load fails on the original
config.json. The shim dir holds a rewritten config.json (the backbone as a Qwen3.5-MoE text config)
and symlinks to the original tokenizer, index and safetensors shards; no weight is copied.

    python models/demos/blackhole/drama3/make_shim_ckpt.py /models/drama-3 /opt/ep0/drama3_shim
"""

import json
import sys
from pathlib import Path

FISH_ONLY_TEXT_KEYS = (
    "mlp_layer_types",
    "rotary_interleaved",
    "use_qk_norm",
    "layernorm_zero_centered_gamma",
    "shared_expert_gate",
)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    full = json.load(open(src / "config.json"))
    text = {k: v for k, v in full["text_config"].items() if k not in FISH_ONLY_TEXT_KEYS}
    text["model_type"] = "qwen3_5_moe_text"
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": text,
        "tie_word_embeddings": False,
        "eos_token_id": full["eos_token_id"],
        "pad_token_id": full["pad_token_id"],
        # Drama 3 extras, unused by qwen36 but kept for the drama3 code
        "drama3": {
            k: full[k]
            for k in ("semantic_start_token_id", "semantic_end_token_id", "audio_pad_token_id", "speech_template_version")
        },
    }
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    for f in src.iterdir():
        if f.name == "config.json":
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f)


if __name__ == "__main__":
    main(*sys.argv[1:3])
