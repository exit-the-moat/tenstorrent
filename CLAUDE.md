# Exit the Moat: public release repo

Code, benchmarks and results for the YouTube channel Exit the Moat (https://www.youtube.com/@ExitTheMoat), so viewers
can reproduce every number in a video. GitHub: https://github.com/exit-the-moat/tenstorrent (**private until
release, then public**: write every line of this repo as if it were public already). Owner: Sebastian.
Video production (scripts, footage, renders, branding, episode plans) lives in a separate private project; this file
holds the porting, optimisation and benchmarking knowledge. **Only published episodes go in here**: no folders,
plans or names of episodes that are not out yet (Sebastian, 2026-10-31).

## Layout

| Folder | Episode | Contents |
|---|---|---|
| `fish-drama3-p150/` | 0 | Fish Audio Drama 3 on one p150: the port (`drama3/`), tt-metal patch, setup, bench scripts, eval lines + scorer |
| `results/` | 0 | `drama3.csv`, `drama3-runs/` (Episode 0 run directories), `rtx-pro-6000-reference/` (the GPU comparison) |

Weights go to Hugging Face (org https://huggingface.co/exit-the-moat), not into git. Episode 0:
`exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150` (private until release).

## Rules

- **Public repo: no secrets, ever.** That covers tokens, API keys, private IPs, internal hostnames and the
  owner's own domains and share paths, SSH details, secret share paths, personal paths beyond the repo, and
  customer data. Check `git diff --cached` before every commit.
- **No private voice recordings.** The voice reference and every take cloned from it stay out of git and out of HF
  uploads unless Sebastian says otherwise for that file.
- Pin and record versions in each README: firmware bundle, tt-kmd, tt-metal commit, container image, driver, date of
  run.
- Every number shown in a video must come from a row in `results/`, produced by code in this repo. Don't edit result
  rows by hand. Re-run instead. Estimates are labelled "estimate" wherever they appear.
- The bench config and the eval gate are not changed to make a number look better. Changes to them get their own
  commit with a reason.
- Licence: Apache-2.0 (`LICENSE`). Third-party patches keep their notices. Mark changed files as Apache-2.0 §4
  requires. `drama3/host/prompt.py` and `host/tower.py` are ports of SGLang-Omni code (Apache-2.0).
- **Pushing, creating the GitHub repo, making it public, uploading to Hugging Face and opening upstream PRs each need
  the user's explicit OK.** Local commits only when the user asks.
- Git identity for this repo: `schoggie <no-reply@exit-the-moat.com>` (repo-local config). Never commit with a personal
  e-mail address; GitHub blocks pushes that expose the private one.
- History: the repo was squashed to one commit before its first release (no unpublished plans, no personal paths in
  history). Keep it clean: check `git log -p` for secrets and later-episode content before making anything public.
- Shell scripts must be committed executable (`git update-index --chmod=+x`); a Windows checkout loses the bit.

## Episode 0: Drama 3 on one p150 (the port)

Status (2026-10-31): the working branch is `ep0/drama3` in a tt-metal worktree on the AI server (base `70222e90` +
the port under `models/demos/blackhole/drama3`, local commits only). The copy in `fish-drama3-p150/` is the export
of that branch at `33145b6b0` (port10 + `DRAMA3_ONLY_BUCKET`, 2026-10-31) plus `--ref none` (branch `b570e31b8`),
with every run directory behind the video (day 3 to port10, load, power, narration). The RTX reference is in
`results/rtx-pro-6000-reference/` (no audio).

**Reproduced from scratch on 2026-10-31** (fresh clone, new container, weights from HF, card 1, `--ref none`): it works;
build 19 min with 8 jobs, first cold run ~10 min (codec compile, backbone build, card codec, warm-up, ~2 min each),
the line itself RTF 0.462 / first audio 0.40 s on the 110-core card. What a stranger needed, now in the README:
relocating `/opt/ep0` (hardcoded in ~90 files: the documented `sed` one-liner, not a symlink on a shared machine),
`TT_VISIBLE_DEVICES` passed through `setup/run.sh`, `setup.sh` no longer needs the `hf` CLI (huggingface_hub from the
Whisper side folder), the lines-file JSON format `[{id, spoken, written}]`, and which red warnings are harmless
(pip dependency conflicts in the Whisper side folder, the tokenizer regex warning). Not done yet (ideas): one root
variable instead of `/opt/ep0`, a `--text` option, a pip constraints file, `TZ` in the container. The full lever-by-lever log is the port's `HANDOFF.md` (newest section first). Re-export it with the
code; don't paraphrase it here.

### What runs where

- Backbone: 40 layers in the Qwen3.5-MoE layout, run on tt-metal's qwen36 model code (30 Gated DeltaNet + 10 gated
  full-attention layers, 256 experts, top-8 + a shared expert). Paged KV, one masked prefill, traced decode.
  Weights: experts gate/up BFP4, down BFP8, projections BFP8.
- Codebook tower (13 distinct layers, 10 codebooks per frame), the sampler, the codec quantizer and the DAC decoder:
  all on the card, traced. Host: tokenisation, prompt, the main-token draw, streaming / playback bookkeeping.
- One frame = 2048 samples at 44.1 kHz = 46.44 ms of audio. RTF = compute s / audio s.
- Code map: `tt/` (backbone_paged, tower_device, codec_device, fused-op wrappers, prefetchers), `kernels/` (hand-written
  Tensix kernels via `ttnn.generic_op`, JIT-compiled, no C++ rebuild), `host/` (`generate_fast.py` solo pipeline,
  `batch.py` static batches, `cbatch.py` continuous batching, `stream_codec.py`, `prompt.py`), `tests/` (golden /
  numerics / gate checks, profiles, benches), `reference/cpu_layer_ref.py` (FP32 CPU reference, one layer in RAM).

### Card facts that matter (measured unless marked)

| | |
|---|---|
| DRAM | 33.9 GB usable in 8 banks, 512 GB/s spec; resident weights 26.5–27 GB |
| L1 (SRAM) | ~1.5 MB per Tensix core; ~180 MB total. Weights stream DRAM → L1 every frame; none stay resident |
| Worker grid | **not identical across cards**: the x16-slot card 12 × 10 = 120 cores, the x4-slot card 11 × 10 = 110 |
| PCIe | Gen3 is enough. Host → card 8.28 GiB/s, card → host 0.77 GiB/s. Cold load (SATA SSD → card) 183–186 s, warm (page cache) 11 s; a Gen5 link would save ~2 s of the 11 (estimate) |
| Host | one CPU, 8 cores / 16 threads, shared by both cards |

### Presets (`run_gen.sh`)

| Preset | What | Reference outputs (must stay identical) |
|---|---|---|
| `--fast` | Day 4b pipeline: paged + traced backbone, prefix cache, device sampling, card codec. Earlier Episode 0 renders used it. **Never changes** | `day4b/out_v2`, 8/8 frames + wavs |
| `--interactive` | `--fast` + every gated single-stream lever (port5–port10), `--tower-mlp bfp8`, fused sampler, backbone + tower weight prefetch. Sebastian approves each addition (by ear + gates) | `port10/i_fin` (2026-10-31 evening) |
| `--batch --batch-size B` | static batches (Day 5b); continuous batching is `host/cbatch.py` with the batch env (port6 `B7`, port10 `C4`) | `port9 q_short_8` for cbatch B 8, base flags |

- Every lever is its own env flag, default off; a preset is a list of flags in `run_gen.sh` (header comment lists each
  addition and its gain; its last line still says bfp8 is not included, the `set --` line is authoritative). Edit
  `run_gen.sh` only while no run uses it, atomically (temp file + `mv`).
- `--interactive` warm-up parks a trace for **one suffix prefill bucket**. A lines file with mixed lengths needs one
  pass per bucket: opt-in `DRAMA3_ONLY_BUCKET=<rows>` (commit `33145b6b0`).
- Current numbers, 8 test lines, voice `ref_v2`, seed 0, card 0 alone:
  - `--interactive`: **RTF 0.381** (lines 0.373–0.412), 17.5 ms per frame, TTFA 0.409–0.472 s, card DRAM 31.29 GB,
    board 162–219 W (mean 195.9), Whisper 0.908 (min 0.744).
  - Batched ceiling (32 jobs at t = 0, all port10 levers, B = 16): short lines **8.05**, long chunks **9.51** audio s
    per wall s.
  - The Episode 0 narration (47 takes in the video): RTF **0.404**, 399 s of speech, first audio median 0.43 s,
    Whisper mean 0.975 (`results/drama3-runs/narration-v20/`, `used_takes.json`, `narr_rtf.py`).
  - `--ref none` = the model's own voice (no reference block in the prompt); otherwise a reference WAV + `.txt`.

### Single-stream levers and what each bought (RTF on the 8 lines unless noted)

| Stage | Lever | Before → after |
|---|---|---|
| Day 3 → 4b | paged KV + traced backbone, tower on card + traced, grouped-MoE prefill, MoE decode K blocking, prefix cache for the voice reference, device sampling, card codec | 14.74 → 0.853 |
| port5 | fused one-token kernels: GDN decode v3, MoE top-k / combine / SwiGLU, attention pre / post SDPA, merged qkv; TTFA levers (first window after 4 frames, voice pre-warm, fused GDN chunk prefill, small-grid suffix prefill) | backbone 23.0 → 12.8 ms/frame; RTF 0.57, TTFA 0.55–0.61 s |
| port6 | grouped-MoE matmul glue, quantizer on the card | TTFA 0.45–0.49 s, playback gapless from the first sound; RTF 0.537 |
| port7 | `DRAMA3_FRAME_PIPE`, `_SAMPLER_LEAN` (both bit-identical), `_TOWER_ETABLE`, `QWEN36_GROUPED_COMPACT` | 0.537 → 0.512–0.517, TTFA mean −10 ms |
| port8 | `DRAMA3_ADDNORM_FUSED`, `_MOE_TOPK_SOFTMAX`, `_MOE_EXPERT_MM` v2 D 8 (8 weight blocks in flight per RISC) | → 0.499–0.506 |
| port10 | codec: dev trace, channel-0 readback, causal pad, fused Snake kernel (bit-identical) | 0.537 → 0.518 |
| port10 | all port7/8 levers + `DRAMA3_MOE_SHARED_IN_EMM` (shared expert as a 9th slot in ExpertMM) | → 0.462 |
| port10 | `--tower-mlp bfp8` (precision call, Sebastian: sounds identical) | 0.463 → 0.440 |
| port10 | `DRAMA3_SAMPLER_FUSED` (one program per codebook draw, 4 cores; bit-identical; 9 draws 1.754 → 0.482 ms) | 0.433 → 0.407 |
| port10 | `DRAMA3_BB_PREFETCH` + `_TOWER_PREFETCH` (DRAM-core tensor prefetcher feeding the decode matmuls; +2.56 GB DRAM; single stream only) | 0.412 → 0.381 |

### Batch levers (port10, full queue, B 16 short lines)

| Lever | Effect |
|---|---|
| B > 8 kernels (GDN v4/v5, router top-k b2, attention post b2, k/v shard grid over rows) | B ≤ 8 → B ≤ 24 runs |
| `DRAMA3_MOE_BATCH_DEDUP` (each unique routed expert read once for all rows; list built on the card, trace-safe) + `_DEDUP_MASK` (idle slots add no experts) | MoE block 1.45 → 0.75 ms; backbone 70.8 → 35.7 ms/step. Changes fp32 summation order, so frames differ: gated, not bit-identical |
| `DRAMA3_CB_VDRAW` (batched host draw) | draw 6.3 → 3.4 ms/step, bit-identical |
| `DRAMA3_KV_COMPACT` (voice-prefix K/V held once) | −0.45 GB at B 16; makes B 24 fit |
| `DRAMA3_STREAM_ENDS=10`, `DRAMA3_CODEC_WINDOWS=32,96` | fewer codec windows; frees DRAM for B 24 |
| codec levers (as above) | codec 16.4 → 7.2 ms/step, audio bit-identical |
| all together (`C4`) | 4.19 → 8.05 (short), 4.93 → 9.51 (long) audio s per wall s |

Ceiling B = 16, set by the 32-job queue (B 24 keeps ~15 slots busy and loses 2–5 %). B 27 runs out of DRAM (640 MiB
grouped-MoE buffer at the voice-prefix prefill); B 32 fails at capture (see gotchas). Admission prefills stall every
stream (~0.4 s each, 17–25 % of the wall).

### What failed or wasn't worth it (measured)

- bfp4 tower MLP: fails the tower check (head top-1 112/180 vs 161 for bfp8) and gains nothing over bfp8; + bfp4
  attention: rejected (top-1 90/180, speaker similarity 0.946).
- HiFi2 DRAM-sharded tower matmuls: 410–440 GB/s in the bench, no gain in the pipeline, +1.1 GB DRAM.
- ExpertMM v1 (port5): slower than sparse_matmul. v2 reaches 229 / 279 GB/s and is bound by the weight reads (math
  off: same time); target 350 GB/s not reached; > 64 cores overflows L1 CBs.
- Day 5 BFP8 expert gate/up on 8 layers: no speed gain (RTF 0.854 vs 0.853), +2.1 GB DRAM; kept BFP4.
- Grouped-MoE bmm program configs (`QWEN36_GROUPED_BMM_CONFIG`): fast and exact in isolation, NaN in the full
  prefill (cause unknown). Off. A two-line gate didn't catch it; the golden check did.
- `DRAMA3_FIRST_WINDOW=2`: lower TTFA, but playback no longer gapless; not in a preset.
- Concurrent codec on spare cores (2 CQs + sub-devices): synthetic 25.9 → 22.0 ms, but the tower / embed traces use
  every core and stock `conv1d` can't be confined to a core subset, so not integrated.
- `Conv1dConfig.config_tensors_in_dram=True`: hung the card twice; removed.
- Single-stream target RTF < 0.173 / TTFA < 0.165 s (the RTX numbers): below the weight-read floor at this precision
  (3.90 GB per frame at 512 GB/s = RTF 0.164 with bf16 tower MLP, before any overhead).

### Quality gates (a lever counts only if it passes)

1. **Bit-identical** where the math is meant to be identical (scheduling, tracing, fusion without reordering): frames
   and wavs 8/8 vs the preset's reference. `--fast` and `--interactive` are re-checked after every change.
2. **Golden check** `tests.check_paged_backbone` vs the FP32 CPU reference (per-layer PCC ≥ 0.99, candidate-logit PCC,
   top-1 at the first frame) after any backbone or prefill change. **Tower check** `tests.check_tower_device` with
   forced codes (layer PCC ≥ 0.999, head PCC, head top-1 out of 180). `tests/check_batch_numerics.py` for B > 1.
3. **Whisper vs script** on the 8 lines: mean no worse than 1 point below the reference; multi-seed close / fix gate
   (seeds 1–5). Whisper misreads "moat" ("mode", even in the original recording) and "Tenstorrent": those words
   need a human ear.
4. **Speaker similarity** (resemblyzer GE2E cosine vs the voice reference) within the take-to-take spread (~0.95).
5. **Spectral bands** within ~0.6–0.8 dB of the reference takes; no clipping; gapless playback (`windows` in timing.json).
6. Precision changes (bfp8 / bfp4) are Sebastian's call after a listen set, never automatic.
7. Fish's sampler is rank-noise based: a tiny logit change changes the draw, so takes differ even when quality doesn't.
   Compare towers with forced codes; one seed is not a length gate.

### Measurement rules

- **One card per model instance, nothing split across cards** (no tensor or pipeline parallel in Episode 0).
- Counted numbers come from the x16-slot card (device 0) **running alone**. A job on the other card, or a Whisper
  check during a run, inflates host time 2–3× (one host CPU). Card 1 is for development only.
- Warm repeats without a page-cache drop between them (a drop causes first-line host stalls). Three repeats for a
  headline number; repeats agree within ±0.003 RTF.
- Every number from a run directory (`timing.json`, `power.jsonl` merged by `power_log.py`, `*.wcheck.log`,
  `spk_*.log`, `spec_*.log`); scoreboard rows by `tests/day7_scoreboard.py`, never typed.
- Power: `tt-smi` sampler, chip and board W, averaged inside each line's window.
- Stop by EOS on every line; report min / median / max per stream for batch runs, TTFA from admission.

### tt-metal / ttnn gotchas learned

- **Never compile while a trace is parked.** Every program shape (MoE M buckets, the prefill bucket) runs once in
  warm-up before `begin_trace_capture`; a new suffix prefill bucket at request time breaks the run.
- **Buffers allocated after a trace capture can overlap that trace's intermediates** and get overwritten by replays.
  Allocate everything that must survive (GDN snapshot, sampler constants, tables) before the capture; read outputs
  from a copy.
- **Persistent L1 reservations change auto program configs**: the prefetcher's global CB made ttnn pick different
  configs for unrelated ops, so takes changed (0/8 identical) although each matmul was bit-exact. Gate it like a
  numerics change.
- **B limits**: the attention pre-SDPA batch kernel uses 4 cores per row, so B 32 needs 128 cores (> 120); DRAM caps B
  at 24–27; the 32-job queue caps the useful B at 16.
- L1_SMALL holds conv1d sliding-window buffers; codec decoders at 3 window sizes ran out of it (`DRAMA3_L1_SMALL`).
- Kernels (Blackhole, this build): `mul_tiles_bcast_scalar` accumulates into DST; dest-reuse binaries truncate fp32
  DST (use fp32 SFPU binaries); zero the whole scalar tile for `sfpu_mul_bcast_col`; DRAM reads of 32 B need 64-byte
  aligned L1 slots; don't name a constant `EPS`; read profiler buffers before the measured piece; stream the inputs
  the compute needs first; avoid every core pulling the same full row (NOC congestion).
- A kernel deadlock or a killed container can leave the card un-openable: a soft reboot clears it. Kill the container
  before releasing a lock.
- Fixed per-program cost dominates one-token decode (~1,100 programs per frame); one row of a 32-row tile does the
  math, 8–64 of 120 cores work. Fusion and fewer programs paid more than faster math.
- Single-device qwen36 MoE needs rank fixes (`[1,1,S,H]` experts); a zero padding row is not neutral for GDN state
  (use valid_len); `ttnn.from_torch(..., bfloat4_b)` without a device still opens the card.
- conv1d truncates fp32 activations to bf16 inside (SNR 31 dB per chain); `ttnn.typecast` to bf16 first rounds to
  nearest (SNR 44 dB). k = 1 convs work better as `ttnn.linear`.

### Benchmarking and the RTX Pro 6000 comparison

- Bench scripts: `fish-drama3-p150/bench/` (`solo.sh`, `ladder.sh`, `power_sum.py`); profiles `tests/profile_frame7.py`,
  `tests/profile_ops_dev.py`; one-op benches `tests/bench_*.py`; 8 test lines `eval/lines.yaml`.
- Reference: the original `fishaudio/drama-3` (BF16 + FP8 experts) on the official SGLang-Omni server, all defaults,
  one rented RTX PRO 6000 Blackwell, same 8 lines and voice: RTF 0.173, TTFA 0.165 s, 8 concurrent streams
  17.7 audio s per wall s, board 262.9–269.8 W mean, Whisper 0.927. GPU rental needs Sebastian's OK each time.
- State the method differences on screen: the server refuses a seed; a different sampler implementation; client-side
  timing incl. HTTP/SSE; whole-pipeline ms per frame (no backbone/tower split); GPU memory is a reservation; different
  weight precision.
- Today: the RTX is 2.2× faster per frame (17.5 vs 8.0 ms) and ~2.5× to first audio. Against each card's own
  weight-read floor (weight bytes per frame / bandwidth / ms per frame): p150 ~41 % (3.69 GB at 512 GB/s), RTX ~37 %
  (~5.3 GB at 1,792 GB/s): **both estimates** from checkpoint shapes × dtypes.
- Ideas left (estimates): multi-sequence suffix prefill for admission, async codec readback or ping-pong half batches,
  a 160-frame codec window, a multi-row attention pre kernel for B > 27.
