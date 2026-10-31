# Drama 3 port: notes (Days 1–3)

Sources (shallow clones in `/opt/ep0/ref/`): SGLang-Omni `sglang_omni/models/fishaudio_drama3/` (main reference) and
vLLM-Omni `vllm_omni/model_executor/models/fish_omni/` (same model, agrees on the points checked below).

## Golden reference (Day 1)

`/opt/ep0/golden/`: `embedding.pt`, `layer_00.pt` … `layer_39.pt` (residual stream after each layer, FP32 [42, 2048],
plus the router's top-8 expert ids/weights per token), `final_norm.pt`, `logits_last.pt` (FP32 [252288]),
`manifest.json` (prompt ids, shapes, sha256 of every file, per-layer rms/absmax, argmax at every position).
`reference/cpu_layer_ref.py --selfcheck 4` shows the layer loop equals transformers' `Qwen3_5MoeTextModel` bit for bit
(layers 0–3 + final norm, max diff 0).

The residual stream grows large: rms 0.58 after layer 0, 53 after layer 38, **501 after layer 39** (absmax 23,926).
BF16/BFP8 on the device will need the residual in a wide enough format; compare with PCC per layer, not abs error.

## Backbone on one P150 (Day 2)

Run (warm cache, about 1 min to build + forward):

    EXTRA_ENV="QWEN36_MOE_MATH_FIDELITY=HiFi2 QWEN36_MLP_MATH_FIDELITY=HiFi2 QWEN36_ATTN_MATH_FIDELITY=HiFi2" \
        models/demos/blackhole/drama3/run_chain.sh /opt/ep0/day2/hifi2all bf16

- Tensor cache: `/opt/ep0/cache/drama3/P150/tensor_cache_bfp8`, 700 files, 25.0 GB (experts gate/up BFP4,
  down BFP8, projections BFP8, norms/router BF16, shared expert gate/up BFP4 + down BFP8 as qwen36 does).
  Filled by `convert_batches.sh` (teacher-forced check per layer + disk guard); sha256 list in
  `/opt/ep0/day2/cache.sha256`. A warm load skips the 23 GB of FP8 experts (`experts=False`) but still reads the
  small per-layer tensors, because qwen36's GDN conv weights are not cached (`load_conv_weight`).
- Card DRAM with all 40 layers + final norm + 4,097-row BF16 candidate head: 25.96 / 34.18 GB.
- The residual PCC is dominated by one massive channel (1269, about 2,800 on the last row; final-norm gain 0.027).
  Compare in the final-norm space (`tests/analyze_chain.py`) to see the error the head sees.
- Precision: with qwen36's default LoFi the top candidate at the last position is wrong (2761 instead of 2499).
  HiFi2 on all prefill matmuls fixes it. The CPU run with the device weight formats and FP32 math
  (`cpu_layer_ref --quant qwen36`, `/opt/ep0/day2/sim_bfp4`) keeps 2499 on top (4/5 overlap), so BFP4 experts are
  not the limit; math/activation precision is. Per-config numbers: `/opt/ep0/day2/summary.json`.
- ttnn host-side BFP conversion (`from_torch` to bfloat4_b without a device) still opens the card and takes the
  KMD lock; a device job started meanwhile waits.

## End to end (Day 3)

    models/demos/blackhole/drama3/run_gen.sh /opt/ep0/day3/gen_all.log --lines all --out /opt/ep0/day3/out

- Backbone on the card (prefill HiFi2, decode qwen36's LoFi); codebook tower, sampling (Fish defaults: T 1.0,
  top-p 0.9, RAS, codebook top-k 30) and codec on the host CPU (`host/`). The codec needs
  `/opt/ep0/pyext-codec` on PYTHONPATH (descript `dac` layers + torchaudio, both `--no-deps`); the Whisper check
  runs in its own process with `/opt/ep0/pyext-whisper` (its huggingface_hub/tokenizers would shadow transformers').
- Prompt: the 718..734 rows (569 semantic rows of the 26.4 s reference) are prefilled up to the largest multiple of
  32 (704); the tail goes through single-row decode, so no padding row enters the KV cache or the GDN state
  (a zero row is not neutral for GDN: beta = sigmoid(0) = 0.5).
- Mixed-embedding golden (`/opt/ep0/day3/golden_voice`, prompt + 24 teacher frames): prefill per-layer PCC min
  0.99746; the last prompt row (the first frame's logits) matches top-1 with 4/5 overlap. Decode HiFi2 vs LoFi:
  no measurable difference (24 teacher frames: logits PCC mean 0.9972 vs 0.9973), so decode stays LoFi.
  Two special-token rows of the prompt tail (`<|im_start|>`, `\n` before `assistant`) sit at layer PCC 0.978
  (final norm 0.95). Day 4 shows the same dip when they are prefilled, so it is specific to those rows in this
  voice context, not a decode-path effect (corrected); the rows that sample frames are unaffected.
- Speed (steady state, the first line also pays the decode-kernel compiles): RTF about 12.5-14; per frame
  about 185 ms on the card (eager decode, 40 layers, readbacks) + 270 ms host tower; first audio about 20 s
  (card prefill 10 s + tail 5 s + tower prefill 4 s). Card about 45 W chip / 165 W board while speaking.
- Fish's default sampler can draw residual codes >= 1024 that the codec's codebooks lack; masked here as Fish's
  reference sampler does.

## Speed (Day 4)

    models/demos/blackhole/drama3/run_gen.sh <log> --fast --lines all --out /opt/ep0/day4/out   # host/generate_fast.py

- Backbone `tt/backbone_paged.py`: paged KV (fixed shapes: no per-length recompiles, trace-safe), persistent
  in-place GDN state, the whole prompt in one masked-bucket prefill (bucket = 128-multiple, GDN valid_len), decode
  as one trace per frame (inputs staged into persistent buffers; bit-identical to eager).
- MoE prefill: qwen36's tile-sparse prefill streams an expert's weights once per 32-token tile and runs down_proj
  for all 256 experts on every row (0.3 s/layer). `qwen36/tt/moe/prefill_grouped.py` groups tokens by expert
  (router top-k read back once per layer) and runs one batched matmul per projection: 10.7 s -> 2.0 s per prompt.
  Expert load is very uneven (one expert gets ~580 of ~720 rows in some layers), so M is bucketed
  (128..768) and every bucket is compiled before the decode trace is parked.
- MoE decode: the sparse_matmul program config streams K one tile at a time (in0_block_w = 1): 0.48 ms / 0.20 ms
  for gate_up / down with 8 active experts. K blocks 32 / 8 with an fp32-accumulating config:
  0.08 / 0.09 ms at identical PCC (limited by the BFP4 weights). Backbone decode 65 -> 45 ms/frame.
- Tower `tt/tower_device.py`: on the card next to the backbone (+1.7 GB), Qwen36GatedAttention with
  de-interleaved rotary rows; 9 traced segments per frame, host sampling between them; frames bit-identical
  eager vs traced. Its sampled codes differ from the host FP32 tower's: Fish's sampler draws its noise per sorted
  rank, so any near-tie swap among the top 30 changes the draw (head top-1 agrees 168/180).
- Codec `host/stream_codec.py`: Fish's stream windows in a worker process (a thread costs the tower step ~20 ms
  through CPU/GIL contention), decoder torch.compile'd (1.6x, identical output). About 30 ms of CPU per frame
  plus the 32-frame overlaps; it keeps up with generation; the last window is a 2-3 s tail per line.
- Every program must be compiled before a trace is captured (generate_fast warm-up: grouped-MoE buckets,
  prefill bucket, tower prefill + step, then the backbone and tower captures).

## Prompt template ("reference-bracket-v1", `prompt.py`)

Rows are `[token id, code 0..9]` (11 ints). Text rows have zero codes. Each literal is tokenized on its own:

    <|im_start|>system\n  +  "convert the provided text to speech"            (no reference)
                          or "convert the provided text to speech reference to the following:"
                          + per reference i: "\n\n[Reference i]\nText: <|speaker:0|>transcript\nSpeech: " + audio rows
    <|im_end|>\n  <|im_start|>user\n  "<|speaker:0|> " + text  <|im_end|>\n  <|im_start|>assistant\n<|voice|>

Audio rows of a reference: token = semantic_start (248,099) + code0, then the 10 codes. The golden prompt is the
no-reference form, 42 tokens. Voice cloning (Day 6) = the reference form with the codec-encoded clip of my voice.

At the end of the golden prompt the backbone puts 88.6 % of the probability on the semantic band, 11.3 % on the
text token `[` (Fish's bracket tags, e.g. `[laugh]`), eos 3e-6. Fish only ever samples among eos + the 4,096
semantic ids, so the first frame would be `<|semantic:2499|>` (greedy).

## Backbone input embedding (`AudioDecoder.embed_backbone`)

    text row:      embed_tokens[id]
    semantic row:  (embed_tokens[id] + codebook_embeddings[code1 + 1*4096]) / sqrt(2)     (vq_backbone_codebooks = 1)

`audio_decoder.codebook_embeddings` is [10*4096, 2048]; codebook k lives at rows k*4096 … Every generated frame is
fed back this way, so the device model needs an input-embedding entry point, not token ids only.

## LM head

`lm_head` is untied. Fish slices it to the 4,097 candidate rows (eos 248,046 + 248,099..252,194) at load time.

## Audio decoder = "codebook tower" (`audio_decoder.py`, `decoder_layers.py`)

- 13 dense layers (4 `audio_decoder_mtp_shared` + 9 `audio_decoder_mtp`), dim 2048, 16 heads / 2 KV heads,
  head_dim 256, MLP 6144 SwiGLU, Gemma-style (1 + w) RMSNorms, gated attention (`wqkv` = [q | gate | k | v],
  Fish re-interleaves q/gate per head on load), **interleaved** RoPE (GPT-J style) on 25 % of the head, base 1e7.
- **It keeps a KV cache along the sequence**, one per tower layer (13 extra full-attention caches, layer ids 40..52
  in SGLang's pool), it is not a per-frame stateless MLP. Each tower layer at position p attends to all earlier
  positions of that same layer.
- One frame step: backbone final-norm hidden h (the lm_head input) → sample main token (semantic or eos) →
  code0 = main − 248,099 → shared layers 0..3 (layer 0 also gets `shared_input` = sum of codebook_embeddings of
  codes 2..9 of the *input* frame) → residual layer r (r = 0..8) mixes `token_embedding(code_r)` → head r gives
  logits for code r+1 → sample. Each layer: `eh_proj(cat(enorm(emb), hnorm(h)))` → pre-norm attention + MLP →
  `final_layernorm`. Non-semantic (eos) labels zero the embedding.
- Prefill: the tower cache of every prompt row is filled too (row p uses frame p+1 as its "next frame").
- Sampling: semantic draw with top-k/top-p/temperature plus **RAS** (if the draw repeats a main token from the last
  16 frames, redraw at T=1.0, top-p 0.95, top-k 30); codebook heads top-k ≤ 30.

## Codec (`codec.pth`, `fishaudio_s2_pro/fish_speech/models/dac/modded_dac.py`)

Fish "modded DAC", 44.1 kHz. `from_indices(codes [B, 10, frames])` → `quantizer.decode` → `decode` → waveform;
`frame_length = hop_length * 4`, one frame per frame_length samples (≈ 21.5 frames/s at 44.1 kHz, i.e. 2,048 samples). The same codec encodes the voice-reference clip
(`encode`). SGLang decodes in streaming windows; for Day 3 a plain CPU `from_indices` of the whole clip is enough.
