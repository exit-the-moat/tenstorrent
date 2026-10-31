# Drama 3 → qwen36 weight mapping (Episode 0, Day 1)

Checkpoint: `fishaudio/drama-3` (`/opt/llm/hf/drama-3`, `FishOmniModel`, `model_type: fish_omni`),
62,245 tensors in 15 shards. Target: `models/demos/blackhole/qwen36` at 70222e90, single P150 (`(1,1)` mesh).

The backbone is a plain Qwen3.5-MoE (Qwen3.6-35B-A3B layout). Fish's own servers load it with the
stock Qwen3.5-MoE model under its Hugging Face names (SGLang-Omni `Drama3Backbone(Qwen3_5MoeForCausalLM)`,
prefix `model.language_model.` stripped), so names map 1:1 onto qwen36's existing remap. Checked on CPU by
`tests/check_mapping.py` (qwen36's `Qwen36ModelArgs` + `remap_qwen36_state_dict` + `_stack_per_expert`).

## Weight groups

| Drama 3 checkpoint group | count | dtype | qwen36 key (single device) | status |
|---|---|---|---|---|
| `model.language_model.embed_tokens.weight` [252288, 2048] | 1 | bf16 | `tok_embeddings.weight` | 1:1; vocab 252,288 (see patches) |
| `lm_head.weight` [252288, 2048] | 1 | bf16 | `output.weight` | 1:1; untied; see patches (candidate rows) |
| `model.language_model.norm.weight` [2048] | 1 | bf16 | `norm.weight` | 1:1 (zero-centred, `1 + w`) |
| `layers.N.input_layernorm` / `post_attention_layernorm` [2048] | 40+40 | bf16 | same | 1:1 (zero-centred) |
| `layers.N.linear_attn.in_proj_qkv.weight` [8192, 2048] | 30 | bf16 | `linear_attn.qkv_proj.weight` | 1:1 rename (qwen36 remap) |
| `layers.N.linear_attn.conv1d.weight` [8192, 1, 4] | 30 | bf16 | `linear_attn.{q,k,v}_conv.weight` [2048/2048/4096, 1, 4] | split by qwen36 remap (Q=K=16×128, V=32×128, same as 35B-A3B) |
| `layers.N.linear_attn.in_proj_{z,a,b}`, `out_proj`, `A_log`, `dt_bias`, `norm.weight` | 30 each | bf16 | same | 1:1 (`norm` is the gated RMSNorm, plain `w`, not `1 + w`) |
| `layers.N.self_attn.q_proj` [8192, 2048] (query + output gate, per-head interleaved) | 10 | bf16 | same | 1:1 |
| `layers.N.self_attn.{k,v}_proj` [512, 2048], `o_proj` [2048, 4096], `q_norm`/`k_norm` [256] | 10 each | bf16 | same | 1:1 |
| `layers.N.mlp.gate.weight` [256, 2048] (router) | 40 | bf16 | `mlp.gate.weight` | 1:1 |
| `layers.N.mlp.shared_expert.{gate,up,down}_proj.weight` [512,2048]/[2048,512] | 40 each | bf16 | same | 1:1 |
| `layers.N.mlp.shared_expert_gate.weight` [1, 2048] | 40 | bf16 | same | 1:1 |
| `layers.N.mlp.experts.E.{gate,up,down}_proj.weight` + `.weight_scale_inv` | 3×10,240 + 3×10,240 | **fp8 e4m3** + fp32 scales per 128×128 block | `mlp.experts.E.*` → stacked by qwen36 `_stack_per_expert` to `gate_up_proj` [256, 1024, 2048] (gate rows first) / `down_proj` [256, 2048, 512] | needs dequant (qwen36 `dequant_fp8_block`, BF16) before the stack; done per layer by `drama3/tt/weight_mapping.py` |
| `audio_decoder.codebook_embeddings.weight` [40960, 2048] (10 codebooks × 4096) | 1 | bf16 | — | **audio side**: backbone *input* embedding for semantic frames (host, Day 3) |
| `audio_decoder_mtp_shared.layers.0..3.*` (eh_proj, enorm, hnorm, semantic_embedding, transformer_layer.{attention.wqkv/wo/q_norm/k_norm, attention_norm, ffn_norm, feed_forward.w1/w2/w3}, final_layernorm) | 4 layers | bf16 | — | **audio side**: codebook tower, shared layers |
| `audio_decoder_mtp.layers.0..8.*` (same + codebook_embedding) and `audio_decoder_mtp.heads.0..8.weight` [4096, 2048] | 9 layers + 9 heads | bf16 | — | **audio side**: codebook tower, residual layers + heads for codebooks 1..9 |
| `codec.pth` (separate file, 1.9 GB) | — | — | — | **audio side**: Fish modded DAC codec (frames → 44.1 kHz audio), host CPU |

No backbone tensor is left unmapped and no key outside `model.language_model.*`, `lm_head.weight`, `audio_decoder*` exists.

## What needs a patch (Day 2)

1. **Config (`fish_omni` unknown to transformers).** `Qwen36ModelArgs` → base `ModelArgs._set_hf_params` →
   `AutoConfig` fails on `model_type: fish_omni`. Today: `make_shim_ckpt.py` writes a `qwen3_5_moe` config.json
   (text_config = Drama 3's minus the fish-only keys) with symlinks to the original shards/tokenizer, at
   `/opt/ep0/drama3_shim`. With it, `Qwen36ModelArgs(mesh_device=None)` parses everything right (40 layers,
   full attention at 3,7,…,39, 256×top-8 experts, I=512, shared 512, partial rotary 0.25 / θ=1e7, eps 1e-6).
   The fish-only keys are all at the values qwen36 hard-codes (`layernorm_zero_centered_gamma`, `use_qk_norm`,
   `attn_output_gate`, no interleaved rotary, gated shared expert, every layer MoE).
2. **FP8 loader reads the whole checkpoint.** `Qwen36ModelArgs.load_state_dict()` sees `*_scale_inv` and calls
   `load_qwen36_state_dict_fp8`, which `get_tensor`s all 62,245 tensors (incl. audio) and dequantizes everything in
   RAM (~70 GB BF16: does not fit in 62 GB), and it keeps the GDN weights RAW for the TP loaders, while the
   single-device GDN loader (`gdn/weights.py`) needs `qkv_proj` + split `q/k/v_conv` (it raises on raw). Patch:
   per-layer loading through `drama3.tt.weight_mapping.load_drama3_state_dict` (key-selected reads, dequant,
   then the single-device `remap_qwen36_state_dict`), and the device conversion layer by layer into the tensor
   cache, freeing each layer before the next. Once the cache exists, `ttnn.as_tensor` cache hits never touch the
   checkpoint.
3. **Vocabulary 252,288 vs 248,320.** Nothing in qwen36 hard-codes 248,320: `vocab_size` comes from the config,
   252,288 is a tile multiple (padded_vocab_size = 252,288 on one device), the single-device LM head is
   replicated, so no shape patch. Fish's special tokens start at 248,044 (`<|pad|>`, `<|im_start|>` 248,045,
   `<|im_end|>`/eos 248,046, `<|voice|>` 248,081, `<|speaker:0|>` 248,083, …) and the 4,096 `<|semantic:i|>`
   tokens are 248,099..252,194, so the semantic band runs past Qwen3.6's 248,320. Two behavioural differences matter:
   - Fish keeps **only the candidate rows** of `lm_head` (eos 248,046 + the 4,096 semantic ids = 4,097 rows) and
     samples among those. On device this can be a 4,097(→4,128 padded)-row LM head: 61× less LM-head weight
     traffic per frame (≈1 GB → 17 MB in BF16) and no 252k-wide sampling. The full-vocab head stays for the
     PCC check against the golden logits.
   - The backbone **input** is not always `embed_tokens`: for semantic frames (voice-reference audio and every
     generated frame) it is `(embed_tokens[main] + codebook_embeddings[code1 + 4096]) / sqrt(2)`
     (`embed_backbone`, `vq_backbone_codebooks = 1`). Text rows use the plain embedding. qwen36 must take
     host-computed input embeddings (or a device gather + add) instead of token ids for those rows.
4. **Single-device MoE path.** On `num_devices == 1` qwen36 picks the non-TP modules for attention/GDN/norms
   (the path validated with the 9B), and for MoE `ReplicateTensorToMesh` + all 256 experts per device
   (`load_expert_weights`, `tp == 1`: no expert sharding, no reduce-scatter). That combination (MoE + `(1,1)`
   mesh) is not validated upstream (35B-A3B is only validated on P150x4). Expert weights per layer on one card:
   gate+up BFP4 ≈ 302 MB, down BFP8 ≈ 285 MB → ≈ 23.5 GB for 40 layers (the plan's estimate). Things to watch on
   Day 2: `Qwen36Experts` prefill asserts `seq_len % 32 == 0` (the golden prompt is 42 tokens, so the model must
   pad to 64), the `prefill_sparsity` sized for 256 local experts, and DRAM headroom.
5. **Dequant precision.** qwen36's `dequant_fp8_block` multiplies in BF16 (rounds the scaled weight to BF16, max
   rel. error ≈ 2^-8); the golden reference dequantizes in FP32. Irrelevant next to BFP4/BFP8, noted for PCC.
6. **MTP head.** Qwen3.6's `mtp.*` spec-decode drafter does not exist in Drama 3; `has_mtp` must be off
   (`enable_mtp` defaults off for MoE configs). Drama 3's `audio_decoder_mtp*` is not a drafter, it is the
   codebook tower.

## Run it

    /opt/ep0/run.sh "python models/demos/blackhole/drama3/make_shim_ckpt.py /models/drama-3 /opt/ep0/drama3_shim"
    /opt/ep0/run.sh "HF_MODEL=/opt/ep0/drama3_shim python models/demos/blackhole/drama3/tests/check_mapping.py"
