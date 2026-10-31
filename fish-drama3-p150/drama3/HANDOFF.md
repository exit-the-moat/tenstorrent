# Drama 3 on one Tenstorrent p150 — handoff (Day 7 freeze, 2026-10-30; started end of Day 4b)

**Update 2026-10-31 afternoon: `--interactive` changed (Sebastian).**
- It now includes the codec levers, all port7 / port8 same-precision levers and `DRAMA3_MOE_SHARED_IN_EMM=1` (commits 9cfa0034c, 9cbcb5599, a4a1bab54).
- RTF 0.537 -> 0.462, TTFA 0.417-0.472 s.
- Later the same day: + `--tower-mlp bfp8` (Sebastian, by ear: identical). RTF 0.440, TTFA 0.415-0.464 s. Whisper 0.908, speaker 0.955, bands within 0.81 dB.
- Evening: + the fused sampler (bit-identical) and the backbone + tower DRAM weight prefetch (Sebastian, by ear: good). Card 0 RTF 0.381, TTFA 0.409-0.472 s, +2.56 GB DRAM. The current reference for `--interactive` is `/opt/ep0/port10/i_fin` (= `k_on_1`).
- Earlier reference for `--interactive`: `/opt/ep0/port10/i_bf8` (= `i_all4`), no longer `port6/i_qdt`. `--fast` is unchanged: `day4b/out_v2` (re-checked: `port10/r_def_bf8`).
- Gate vs i_qdt: Whisper 0.908 = 0.908, speaker 0.953 vs 0.948, bands within 0.34 dB.

## Day 8e / port10 (2026-10-31): batched throughput, B up to 24. Read this first

**Status.** Every lever is opt-in and committed locally on `ep0/drama3` (commit ebfac5d3b and the codec follow-up commit). Nothing was pushed. Defaults are unchanged:
- `--fast` is 8/8 identical to `day4b/out_v2`, and `--interactive` 8/8 to `port6/i_qdt` (`/opt/ep0/port10/r_def_10a`, `i_base_10a`, `defs_10b.log`).
- cbatch B 8 with the base flags: frames 32/32 identical to port9 `q_short_8`.

The results are summarised in `fish-drama3-p150/README.md`; the run files are in `results/drama3-runs/port10/`.

**Result** (full queue: 32 jobs at t = 0, greedy, card 0 alone), in audio s per wall s:

| workload | base B 8 | C4 at B 8 | B 12 | B 16 | B 24 |
|---|---|---|---|---|---|
| short (8 lines x 4) | 4.19 | 6.19 | 7.04 | **8.05** | 7.86 |
| long (narration chunks x 4) | 4.93 | 7.01 | | **9.51** | 9.00 |

Quality vs base: Whisper within 1 point or better, no clipping, bands within 0.65 dB, speaker similarity unchanged.

**Ceiling: B 16.**
- With 32 jobs, B 24 keeps only ~15 slots busy, and the idle slots still cost per-step work.
- DRAM is 31.8 of 33.9 GB at B 24, and that needs `DRAMA3_KV_COMPACT=1` plus `DRAMA3_CODEC_WINDOWS=32,96`.
- Without them, B 24 runs out of memory: a 640 MiB grouped-MoE prefill intermediate, or an untilize during the codec capture.
- Measured: B 27 runs out of DRAM at the voice-prefix prefill (port10/lim.log). B 32 fails at capture: the attention pre kernel (4 cores per row) needs 128 cores; card 0 has a 12 x 10 grid, card 1 has 11 x 10.

**C4** = B7 (port6 batch env) + these flags:

`DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1 DRAMA3_KV_COMPACT=1 DRAMA3_STREAM_ENDS=10 DRAMA3_CODEC_DEV_TRACE=1 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1 QWEN36_GROUPED_COMPACT=1 DRAMA3_CODEC_WINDOWS=32,96`

**Levers**
- B > 8:
  - GDN v4 runs to B 16 (both tile halves). `kernels/gdn_decode5_*` cover B 17-32: the partial sums sit in two tile rows (page + 256).
  - `router_topk_b2`, `attn_post_b2_*` (rows per core). The k/v shard grid spans grid rows.
  - cbatch row copy: `d0 = (2b // 32) * nq`, `dr = 2b % 32`.
- `DRAMA3_MOE_BATCH_DEDUP=1`: ExpertMMB (`kernels/expert_mm3_*`; `expert_mm4_writer` for B > 16 drains outputs while the weights stream).
  - Every RISC builds the unique-expert list (first-occurrence order) from the router's B*K ids at run time; the compute gets the count via `read_tile_value`.
  - Core = (n tiles, slot group). `MoECombineB3` / `B4` (B > 16: down tiles in chunks of 16).
  - MoE block 0.78 -> 0.48 ms (B 8), 1.45 -> 0.75 ms (B 16).
  - Its output is identical to the fused path on one layer. Real runs still diverge in frames: the merged slot order changes fp32 sums. Gated.
- `DRAMA3_MOE_DEDUP_MASK=1`: an active-row mask (DRAM `[1, 32]` uint16, `set_active_rows`). Idle slots add no experts.
- `DRAMA3_CB_VDRAW=1`: batched host draw, bit-identical.
- `DRAMA3_KV_COMPACT=1`: `kv_pool_blocks`; the voice prefix (`DRAMA3_KV_PREFIX_BLOCKS`, default 10) is held once.
- `DRAMA3_STREAM_ENDS=10`: windows 10, then +64.
- Codec (subagent):
  - Flags `DRAMA3_CODEC_OUT_CH0`, `_CAUSAL_PAD`, `_SNAKE_KERNEL` (`kernels/snake_*`), each with `DRAMA3_CODEC_DEV_TRACE`.
  - 96-frame window 83 -> 40 ms, bit-identical.
- `DRAMA3_CB_THREADS`: no gain.

**Pitfalls**
- Host CPU contention: two card jobs at once, or Whisper during a run, inflate host ms 2-3x. Count numbers only from card-0 runs that run alone.
- Card locks: `/opt/ep0/card0.lock` and `card1.lock`; the old `card.lock` means all cards (see `port10/env10.sh`). Card 1 = `TT_VISIBLE_DEVICES=1`, slot 2, Gen3 x4.
- `Conv1dConfig.config_tensors_in_dram=True` hung the card twice; it was removed. Killing a hung container can leave the card un-openable, and a soft reboot clears it.
- A `t1.sh` timeout releases the lock even while the container is still running: kill the container first (`port10/codec/t1c.sh`).

**Next** (estimates):
- A multi-sequence suffix prefill: admission is 17-25 % of the wall.
- Async codec readback, or two half-batches, to hide the ~5-10 ms/step of host time.
- A 160-frame codec window.
- A longer queue plus more DRAM headroom for B > 24.

## Day 8d / port8 (2026-10-31): the remaining levers of the port7 list, measured. Read this first

**Status.** All levers are opt-in. Commit 0f9e4c102 on `ep0/drama3`, local only. `run_gen.sh` was not edited.
- Defaults re-checked: `--fast` 8/8 identical to `day4b/out_v2` (`/opt/ep0/port8/r_def`), `--interactive` 8/8 identical to `port6/i_qdt` (`port8/i_base`).
- dmesg I/O errors 0 throughout, disk 70 %, `smoke.py` SMOKE_OK at the end (`port8/smoke_end.log`).
- One soft reboot at 03:15 (`port8/events.log`). The first ExpertMM v2 down-path kernel deadlocked: the A ring was 2 jobs, but the weight stream ran D blocks ahead. The killed container left the card hung on the next open. Fixed: A is now held per slot run, with a 3 KT ring.

**Levers** (frame times: `tests/profile_frame7 --mode time`, `port8/ft_*.json`; base = port7 flags `P7F`: backbone trace 12.02 ms, tower 7.41 ms):
| lever (env) | before -> after | checks |
|---|---|---|
| `DRAMA3_ADDNORM_FUSED=1` (`tt/addnorm_fused.py` AddNormFused2, `kernels/addnorm2_*`): one program per one-token `h = x + d; y = norm(h) * w` on 8 cores. Core k packs its 8 row-0 chunks of x and d into one tile, adds (bf16), squares, sums (`ones @ (sq @ ones)`). It writes row 0 of its partial-sum tile into row k of G on all 8 cores (32-byte writes + semaphore increments), then `S = ones @ G`, rsqrt, and writes its 8 tiles of h and y. Backbone: `_decode_body_addnorm` (Qwen36DecoderLayer.forward, single device, decode, with the 81 boundaries fused, the final norm included). Tower: `TowerLayerDevice._call_addnorm` (3 add+norm + 1 plain norm per layer) | `tests/check_addnorm.py` (`check_addnorm2.json`): chain of 40 boundaries, add + sharded norm 12.22 -> 7.75 us per boundary; norm only 7.82 -> 6.08. port5-style v1 (every core packs the whole row): 15.65. Frame: backbone 12.02 -> 11.81, tower 7.41 -> 7.28 ms | h bit-equal to ttnn.add (8 trials incl. the 2800 channel); y PCC vs torch 0.999995-1.0 (sharded norm 0.999992). Golden (`golden_an.json`): logits PCC 0.99746 (port6 0.99749), top-1 19/24, final norm min 0.98531 (0.98593). Tower check (`tower_an.log`): layer PCC min 0.99912 (et 0.99893), head PCC min 0.99150 (0.99427), top-1 165/180 (161). `DRAMA3_ADDNORM_H_L1=1` (sum in L1): no gain (11.64 vs 11.61) |
| `DRAMA3_MOE_TOPK_SOFTMAX=1` (`kernels/router_topk_sm.cpp` + `router_softmax_compute.cpp`): the top-k kernel hands the score tile and a max tile to a compute kernel on the same core. `E = exp(X - M)`, `S = E @ ones`, `V = E / S`; padding -32768 | backbone 11.81 -> 11.61 ms (`ft_an_sm`) | golden (`golden_sm.json`) logits 0.99761, top-1 19/24 |
| `DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8` (`kernels/expert_mm2_*`): port5's ExpertMM, but each data-movement RISC keeps D weight blocks in flight (trid = 1 + b % D, a ring of C = 2D blocks; `reserve_back(D * BLK)` == "slot b % C is free"). Outputs are written at the end (the output CB holds all jobs), and A is held per slot run | backbone 11.61 -> 11.505 (D 4) / **11.379 (D 8)** / 11.387 (D 12) / 11.38 (D 8 KB 32). v1 (port5) in golden: 13.7 vs 12.8 ms/step | golden (`golden_emm2_d4.json`) identical metrics to sparse_matmul (logits 0.997462). Device profile (`prof_sm.json` vs `prof_s8.json`): gate_up 44.6 -> 41.2 us (9.44 MB: 212 -> 229 GB/s), down 31.9 -> 31.9 us (8.91 MB: 279 GB/s), and the idx to_layout + sparsity glue (2.6 + 3.2 us + gaps) are gone. `DRAMA3_MOE_EXPERT_MM_NO_MATH=1` (`prof_nomath.json`): 40.2 / 31.0 us, so the kernel is bound by the weight reads, not the math. More than 64 cores (80, 96): TT_THROW "statically allocated circular buffers ... beyond max L1" (not investigated) |
| precision candidate `--tower-mlp bfp4` (`host/generate_fast.py`, `tests/check_tower_device.py`; `tt/decode_linear.py` BFLOAT4_B entries = the bfp8 configs, not swept) | 8 lines: RTF 0.507 (default programs, `s8b4`), 0.475 (bfp8 configs, `s8b4c`); bfp8 MLP 0.478 (`s8b8`) | tower check (`tower_b4.json`): layer PCC min 0.97942, head PCC min 0.92977, top-1 112/180 (bfp8 `tower_b8.json`: 0.99902 / 0.99310 / 161). Whisper 0.932 / 0.921, seeds 0.938 / 1.000, speaker 0.955 / 0.955. Fails the tower check. `tests.tune_decode_linear` (TUNE_TOWER_MLP=bfp4) dies in the codec warm-up (L1_SMALL OOM, `tune_b4.log`) |
| precision candidate `DRAMA3_TOWER_ATTN_BFP4=1` (qwen36 `attention/weights.py` PROJ_DTYPE, set only while the tower builds) with bfp4 MLP | RTF 0.517 (`s8b4a`) | tower check top-1 90/180, layer PCC min 0.94863; speaker 0.946 (min 0.923); rejected |

**S8 = P7F + the three levers above** (`render8.sh`, 8 lines, ref_v2, seed 0):
- RTF 0.506 / 0.499 / 0.502 (`s8_1`..`s8_3`, frame-identical to each other).
- TTFA 0.430-0.496 s (means 0.475 / 0.460 / 0.458).
- Whisper 0.908 (min 0.744), seeds close / fix 0.938 / 1.000, speaker 0.953 vs ref_v2.
- `spec_cmp` bands within 0.34 dB of `i_qdt`.
- Device profile (`prof_s8.json`): backbone 591 programs, span 11.56 ms (kernels 10.79, gaps 0.77; port7: 628 programs, gaps 2.34). Tower 527 programs, 7.55 ms (gaps 0.39).
- With bfp8 MLP: RTF 0.478.

**Floor (bytes / 512 GB/s):** 3.90 GB per frame = RTF 0.164. bfp8 MLP 3.44 GB = 0.145. bfp4 MLP 3.20 GB = 0.134 (tower DRAM 1.512 -> 1.267 GB, as computed). + bfp4 attention 3.02 GB = 0.127.

**Stretch** (`host/cbatch.py`, port6 lad3 c2 / c4 env; the port8 levers do not act in batch mode):
- `st_c2`: per-stream decode RTF 0.648-0.719, 2.68 audio s per wall s, 198 W.
- `st_c4`: 0.856-0.968, 2.04 audio s per wall s, 197 W. Mean active streams 1.96 under the guard policy.
- Both equal to port6 lad3.

**Not done / limits:**
- ExpertMM above 64 cores (L1 CB error).
- A real bfp4 matmul sweep (the tuner is broken today).
- Softmax / top-k in the batch path.

Scripts: `/opt/ep0/port8/{env8,chain1..7,gate8,render8}.sh`, `rows.py`, `mk_emm2.py`, `fix_emm2.py`, `mk_sm.py`, `patch_emm2.py`.

## Day 8c / port7 (2026-10-30/31): beat the RTX PRO 6000 on one p150? Not at this precision. Read this first

**Status.** Every lever is opt-in and committed on `ep0/drama3`: e53a1ce08, 03b5bf31b, 6a706ae18 and the amendment commit (HiFi2 tower, compact expert prefill, speaker similarity). Nothing
was pushed. `run_gen.sh` was not edited. The defaults are unchanged:
- `--fast` is bit-identical to `day4b/out_v2`: `/opt/ep0/port7/r_def`, 8 of 8 lines, frames and wavs.
- `--interactive` is bit-identical to `port6/i_qdt`: `port7/i_base7`, 8 of 8.
- Both re-checked after the amendment: `port7/r_def2`, `port7/i_base8` (see `chain13.log`).
- dmesg I/O errors 0 at every step, disk 70 %, `smoke.py` SMOKE_OK at the end.

Target (Sebastian): single-stream RTF < 0.173 and TTFA < 0.165 s, the RTX PRO 6000 reference (`results/rtx-pro-6000-reference`).
**Not reached and not reachable at the current weight precision** (arithmetic below). Every number below is from a run
file in `/opt/ep0/port7` (8 test lines, ref_v2, seed 0) unless the row says otherwise.

**Arithmetic (checkpoint shapes x card dtypes).** A frame is 2048 samples at 44.1 kHz = 46.44 ms, so RTF 0.173 = 8.03 ms
per frame for everything. Weight bytes read per frame:
- Backbone 2.25 GB: GDN 30 x 35.8 MB, attention 10 x 29.0 MB, MoE 40 x 21.7 MB (8 routed experts, bfp4 gate/up and bfp8 down, + the shared expert), candidate head 17 MB.
- Tower 1.65 GB: 13 layers x 121 MB (bf16 eh + MLP, bfp8 attention), heads + one-hot embeddings 71 MB.
- At the 512 GB/s spec bandwidth the 3.90 GB take **7.6 ms = RTF 0.164**, with no codec, no host and no op overhead.
- The tuned one-token matmuls reach 230-440 GB/s, so a realistic floor at this precision is ~11-12 ms per frame (RTF ~0.25, estimate).
- TTFA floor: the ~77-row suffix prefill touches ~233 of 256 experts per layer (~21 GB of expert weights, ~48 ms at peak), plus 4 frames and the first codec window: ~0.11 s at 100 % of peak.

**Frame profile** (`tests/profile_frame7.py`, port5 interactive env; `frame_time.json`, `frame_prof.json`):
| piece | device | notes |
|---|---|---|
| backbone decode trace | 12.02 ms (628 programs: kernels 10.1 + gaps 2.34 ms) | GDN layer ~23 programs ~287 us kernels: in-proj 72, fused GDN 37, out-proj 24, experts 45 + 32 (~230 GB/s), router / top-k / softmax ~20, shared expert ~20, 2 norms (3 programs each) ~15, adds / combine ~11 |
| tower trace | 8.05 ms (631 programs: kernels 6.67 + gaps 1.72 ms) | layer: 32 programs ~430 us kernels, matmuls 317 us at ~380 GB/s, 5 norms x 8.5 us, attention misc ~52 us, adds / muls ~17; head: matmul ~14 us, fp32 top-k ~44 us (matched to the bench by duration), ~20 small sampler programs + one-hot embedding 20 us |
| both back to back | 20.05 ms | |
| pipeline (i_qdt) | 24.65 ms per frame | card codec windows ~1.6-1.8 ms, the prefill amortised ~1.3 ms, card-idle host time ~1-2.5 ms |

Benches (one-op traces, 8.6 us replay floor): `tests/bench_dram_mm.py` (`bench_dram_mm.json`): bf16 at HiFi4 is compute-limited
(~340 GB/s, DRAM-sharded or not); DRAM-sharded at HiFi2 reaches 410-440 GB/s (a precision change); bfp8 one-token matmuls 280-330 GB/s.
`tests/bench_sample_ops.py` / `bench_sample_ops2.py`: fp32 `ttnn.topk` costs 643 us at width 4128 (32 us in bf16), 55 us at width 1024;
`ttnn.gather` ~400-500 us; `ttnn.cumsum` ~93 us at width 64. `tests/count_kept.py` (`kept.json`, 24 lines, 6329 draws): the main-token
top-p 0.9 keeps median 17, max 219 ranks (> 64 in 3.6 %, > 128 in 0.4 %).

**Levers** (each behind its own env flag):
| lever (env) | before | after | gate |
|---|---|---|---|
| `DRAMA3_FRAME_PIPE=1`: the backbone input row is embedded on the card (`BackbonePaged.prepare_embed / capture_embed`: text row staged by the host + `codebook_embeddings[4096 + code 1]` gathered from the tower's code), so the backbone step is queued right behind the tower replay; the codes are read back while the backbone runs (from a copy the embed replay makes: the fused tower trace's own output buffer is allocated after the backbone capture and can be overwritten by the backbone replay) | RTF 0.535 (`i_base7`) | 0.530 (`i_pipe2`) | 8/8 frames + wavs identical to `i_qdt` |
| `DRAMA3_SAMPLER_LEAN=1`: the codebook draw in fewer programs (one keep compare `cum + c <= top_p`, the code as `sum(ids * (score == best))` without the lowest-rank tie break) | 0.530 | 0.524 (`i_lean2`, with the pipe) | 8/8 identical to `i_qdt`; `tests/check_sampler_lean.py` 200 eager + 50 traced rows, 0 mismatches vs the host draw |
| `DRAMA3_TOWER_ETABLE=1`: the e half of eh_proj, `norm_e(emb[c]) @ W_e`, as fp32 host tables rounded to bf16 for every tower layer but shared 0 (shared 1-3 + residual 0 looked up on the host, residual 1-8 by the one-hot matmul on the card); removes 12 one-token norms + 12 matmuls per frame | 0.524 | **0.513-0.518** (`i_et2`, repeats `r4` `r5` `r6`: 0.517 / 0.515 / 0.518) | tower check (`tower_et.log` vs `tower_base.log`): step layer PCC min 0.99893 (0.99908), head PCC min 0.99427 (0.99334), head top-1 161/180 (155/180); Whisper 8 lines 0.915 (`i_qdt` 0.908); seeds 1-5 close / fix 0.938 / 1.000 (= port6 `s_qd`); bands within 0.6 dB of `i_qdt` (`spec_cmp.jsonl`, take-to-take spread 0.4 dB); gapless |
| precision (Sebastian's call): the three above + `--tower-mlp bfp8` (port5's tuned BFP8 tower MLP) | 0.516 | **0.493-0.498** (`i_bf8`, `b5`, `b6`; `b4` 0.512 with one host stall in one line) | tower check bfp8 + etable: layer PCC min 0.99894, head PCC min 0.99176, top-1 155/180; Whisper 0.920; seeds 0.938 / 1.000; bands within 0.6 dB |
| `QWEN36_GROUPED_COMPACT=1` (TTFA): the M = 32 grouped-MoE suffix prefill over the active experts only (146 of 256 per layer on average, `need.json`: 360 suffix layer calls, M = 32 in 195, 64 in 152, 96 in 13), as a compact `[1, A, 32, H]` block through sparse_matmul's gather mode, A bucketed 64..224 (`tests/bench_grouped_compact.py`: gate_up + down 2.60 ms -> 1.48 / 1.82 ms at A = 128 / 160; at M = 64 the gather mode is slower, so it stays off there) | prefill backbone mean 0.326 s, TTFA mean 0.474 s (`r4`-`r6`) | prefill 0.317 s, **TTFA mean 0.464 s, 0.430-0.498** (`c_cmp`, `c_cmp2`, `c_cmp3`; RTF 0.512-0.517) | prefix self-check (`prefix_compact.json` vs `prefix_base.json`): suffix PCC 0.9984 / 0.9982 (0.9983 / 0.9982), cand-logit PCC 0.9969 / 0.9985 (0.9976 / 0.9987), top-1 same; one fix suffix row at PCC 0.951 (0.993); Whisper 0.913, seeds 0.938 / 1.000; speaker 0.955 |
| precision candidate `DRAMA3_TOWER_DS=1`: tower eh_h / w1 / w3 / w2 DRAM-sharded at HiFi2 (bf16 weights; the sharded norm output feeds them directly) | 0.515 (`r5`) | 0.514 (`c_ds`); with bfp8 MLP 0.512 (`c_dsb`, vs 0.493-0.498 without DS) | **no gain in the pipeline** (the bench's 410-440 GB/s did not carry over); tower check layer PCC min 0.99897, head min 0.99420, top-1 159/180; +1.1 GB DRAM (second weight copies); not gated further |
| (measured, not installed) tower MLP `[w3 | w1]` as one matmul + the SwiGLU kernel (`tests/bench_tower_mlp.py`) | 151.8 us per layer | 143.7 us | -0.1 ms per frame |
| (measured, not installed) 2D program configs for the tower prefill's 128-row matmuls (`tests/bench_prefill_mm.py`) | | -125 us per layer | -1.6 ms of the 43 ms tower prefill: the prefill is not matmul-bound |
| (measured, not built) main-token draw on the card | | | the card-idle host time left with the pipe is ~1 ms per frame (`t_pipe` `pipe_ms_per_frame`); fp32 top-k on 4097 logits alone costs 0.64 ms |

TTFA: 0.43-0.50 s in every configuration (`r4`-`r6` 0.449-0.496, `b5`/`b6` 0.444-0.483, `c_cmp*` 0.430-0.498): the suffix
prefill backbone 0.29-0.34 s, the tower prefill 0.043 s, 4 frames ~0.09 s, the first window 0.03 s. Only the compact expert prefill
moved it (-10 ms mean).

Device-side expert grouping (port6 idea: fixed M, ranks on the card, no routing readback): not built. Upper bound from port6's
replay without any readback (`profile_suffix_sync.py`): -24 ms; a fixed M would have to be 96 here (need.json: 42 % of the
suffix layers need M = 64, 4 % M = 96; port6: M = 96 costs ~+17 ms) plus ~6 ms of rank ops, so the net is ~-1 to -10 ms
(estimate). The compact path above took the cheaper part of the same pot.

Speaker similarity (`/opt/ep0/port7/spk.py`: resemblyzer GE2E embedding cosine, installed with --no-deps into
`/opt/ep0/pyext-spk`, 17 MB; `spk.log`, `spk2.log`, per-line `spk_<dir>.json`):
| set | vs ref_v2 mean (min) | vs the i_qdt take of the same line, mean (min) |
|---|---|---|
| port6 `i_qdt` | 0.948 (0.938) | 1 |
| port5 `final8c` (another take, same quality) | 0.951 (0.937) | 0.966 (0.930) |
| day4b `out_v2` | 0.953 (0.943) | 0.970 (0.945) |
| `i_et2` (pipe + lean + etable) | 0.951 (0.940) | 0.968 (0.929) |
| `i_bf8` (+ bfp8 tower MLP) | 0.955 (0.936) | 0.969 (0.934) |
| `c_cmp` (+ compact prefill) | 0.955 (0.942) | 0.970 (0.954) |
| `c_cmpb` (+ compact prefill + bfp8) | 0.960 (0.949) | 0.974 (0.965) |
| `c_ds` / `c_dsb` (HiFi2 tower) | 0.952 / 0.954 | 0.968 / 0.965 |
Every candidate is within the take-to-take spread of the base configuration. bfp4 tower attention / MLP was not measured
(time; the bfp8 step already showed the smallest head PCC).

Tower read count per frame (code, `tt/tower_device.py` `_fused_body`): 13 distinct layers (4 shared + 9 residual, each its own
weights: `audio_decoder_mtp_shared.layers.0-3`, `audio_decoder_mtp.layers.0-8`), each run once per frame, so every tower
weight is read exactly once per frame; nothing is re-read across codebook steps. 1.58 GB of layer weights do not fit the
~180 MB of L1 (120 x 1.5 MB); only the heads (9 x 4.2 MB) or a part of one layer would.

Repeat runs: `r1`..`r3` / `b1`..`b3` ran with a page-cache drop between the runs (`render7.sh`), and the first lines after
each drop had host stalls (RTF up to 0.70, TTFA up to 1.78 s, frames still identical). `r4`..`r6` / `b4`..`b6`
(`render7b.sh`, `NODROP=1`, one drop + sha256 check over all six dirs at the end) are the warm repeats. Power (`power.jsonl`
merged): chip 57-59 W mean, board 187-201 W mean per run.

**What would move it further (estimates, not measured):**
- Dispatch gaps: ~1260 programs per frame at ~3 us each (~4 ms). Fewer programs: one-token norm + residual add as one program (146 norms per frame, 3 programs each), or a whole-layer kernel.
- MoE expert matmuls at ~230 GB/s: ~-1.3 ms per frame at ~400 GB/s (custom kernel, days; port5's ExpertMM showed the weight stream is the hard part).
- The codec windows (~1.6 ms per frame on the same queue): bfp8 / HiFi2 decoder weights would be a quality call.
- TTFA: the grouped-MoE suffix prefill pads 32 rows per expert (8192 rows for ~616 slots); 16-row tiles or a gather-based expert prefill. Plus the GDN chunk op and the per-layer glue.
- Even with all of these, my estimate is RTF ~0.40-0.45 and TTFA ~0.35 s at this precision. To beat 0.173 the bytes per frame must fall below ~3 GB, and the op overhead must go to near zero.

**Proposal for the live preset (Sebastian decides):** add `DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1` to `--interactive`
(bit-identical, RTF -0.01). `DRAMA3_TOWER_ETABLE=1` and `QWEN36_GROUPED_COMPACT=1` change the numbers slightly (gates passed;
RTF -0.01, TTFA -10 ms); `--tower-mlp bfp8` is a precision call (RTF -0.02). Not proposed: `DRAMA3_TOWER_DS` (no gain).

**Files:** `host/generate_fast.py` (pipe loop), `tt/backbone_paged.py` (prepare_embed, capture_embed, read_codes),
`tt/tower_device.py` (enqueue_fused_pre / read_fused, e tables, pre_a, DS), `tt/decode_norm.py` (sharded), `tt/device_sampler.py`
(lean), qwen36 `moe/prefill_grouped.py` (compact); tests `bench_grouped_compact`, `count_need`, `profile_frame7`,
`bench_dram_mm`, `bench_sample_ops`, `bench_sample_ops2`, `count_kept`, `check_sampler_lean`, `bench_tower_mlp`, `bench_prefill_mm`.
Scripts: `/opt/ep0/port7/{env7,chain1..13,gate,render7,render7b}.sh`, `rtf.py`, `spec_cmp.py`, `spk.py`.

## Day 8b / port6 (2026-10-30): first audio and parallel voices. Read this first

**Status.** Every lever is opt-in and committed on `ep0/drama3`: commits 08cfa23fb, b0df02dea, b8a7c9677 and this
write-up. Nothing was pushed. The defaults are unchanged:
- `run_gen.sh --fast` is bit-identical to `day4b/out_v2` after all port6 commits: `/opt/ep0/port6/r_def2`, 8 of 8 lines, frames and wavs, SUMS_OK.
- `run_gen.sh --interactive` is bit-identical to today's base: `port6/i_final` vs `port6/i_base`, 8 of 8. `i_base` is itself bit-identical to `port5/final8c`.
- The golden check with the proposed levers (`port6/golden.json`) equals port5's `golden_sgrid`: prefill layer PCC min 0.99938, final norm 0.99546, last logits 0.99778, decode 12.8 ms/step, logits 0.99749, top-1 19/24.
- dmesg I/O errors 0 at every step, disk 69 %, smoke.py SMOKE_OK at the end. `run_gen.sh` was not edited: the new levers wait for Sebastian.

Every number below is from a run file in `/opt/ep0/port6`, on the 8 test lines, ref_v2, seed 0, unless the row says otherwise.

### Track A: time to first audio (solo)
Start: `--interactive` TTFA 0.573-0.592 s (`i_base`).

| lever (env) | before | after | check |
|---|---|---|---|
| (measurement) the 40 per-layer routing readbacks of the grouped-MoE suffix prefill (`tests/profile_suffix_sync.py`: routing replayed from host memory, no sync) | synced suffix prefill 0.410 s | 0.386 s without any readback | the readback is NOT the big part (the brief assumed it was): 25 ms of 0.41 s. The device time is: E x M padded rows, glue on 8192-row tensors (`tests/bench_grouped.py`) |
| `QWEN36_GROUPED_MM_GLUE=1`: row gather and k-slot combine as two matmuls with [S, E*M] one-hot / routing-weight matrices built on the card by `ttnn.scatter` from [S, K] lists (instead of embedding gathers over row-major copies, multiply, concat, sum); S <= 128 only, so voice prefill and golden are untouched | glue 2.85 ms/layer at M = 32 (bench); synced suffix 0.410 s; pipeline suffix backbone 0.366-0.385 s, TTFA 0.573-0.592 s | glue 1.28 ms; 0.362 s; 0.300-0.339 s, **TTFA 0.508-0.538 s** (`i_glue2`; `i_glue` 0.520-0.548 + first line 0.628) | prefix self-check cand-logit PCC close 0.9976 (base 0.9984) / fix 0.9987 (0.9985), top-1 same, top-10 10 / 9 (`prefix_glue.json`); seeds 1-5 close / fix Whisper 0.938 / 1.000 (`s_glue`) |
| `QWEN36_GROUPED_TOPK_READ=1`: grouped prefill from the router's on-card top-k (no scatter, no host top-k) | 0.362 s synced | 0.355 s; pipeline TTFA 0.505-0.548 s (`i_tk`): **no measurable gain**, not proposed | seeds equal (`s_tk`) |
| `DRAMA3_QUANT_FAST=1`: CPU quantizer op forms (`host/quant_fast.py`: codebook + out_proj as tables, transposed conv as one matmul, depthwise conv as shifts) | 55.2 ms for 4 frames (6 threads, `tq4.log`) | 42.1 ms (rel. err 1e-6); in the pipeline latent wait 0.054 -> 0.043 s, TTFA within noise (`i_qf`) | the post-module transformer alone is 33.6 ms on the CPU |
| **`DRAMA3_QUANT_DEVICE=1`**: the whole quantizer decode on the card (`tt/quant_device.py`: host table sum, then the 8-layer post-module + 2 upsample stages as ttnn ops, traced per window size 32 / 64 / 96), output z stays on the card and feeds the card decoder; windows decoded in the generation loop (`host/stream_codec.py` DeviceQuantStream, no codec process) | first window: latent 0.054 s (CPU) + card decode 0.031 s; TTFA 0.508-0.538 s (`i_glue2`) | quantizer + decoder 0.030 s; **TTFA 0.453-0.490 s** (`i_qdt`), RTF 0.53-0.57 | frames identical; whole-line wav SNR vs the CPU-latent path 37.0-39.8 dB (latent PCC 0.99996, rel. err 0.94 %, `check_quant_device.json`; Day 4b card decoder vs CPU codec: ~44 dB); seeds 1-5 close / fix 0.938 / 1.000 (`s_qd`, eager version); quality call for Sebastian |
| `DRAMA3_FIRST_WINDOW=2` on top of glue (+ quant device) | 0.508-0.538 (0.453-0.490) s | 0.463-0.510 s (`i_glue_fw2`); **0.408-0.455 s** (`i_qdt_fw2`) | but playback from the first sound is no longer gapless: worst chunk 0.109-0.115 s late (`i_qdt_fw2`), so the gapless start is ~0.52-0.57 s, no better than window 4 |

**New metric: gapless playback** (`stream_codec.playback_check`, in every timing.json now as `windows`,
`underruns`, `worst_late_s`). Playback starts at the first chunk. Any later chunk that arrives after playback has
reached it is an underrun. The worst lateness is the pre-buffer that would have made the line gapless. TTFA alone
hides this:
- Fish's windows (10 / 20 / 32 / 64 / 128, ...) need generation at RTF <= 0.5 at their 64 -> 128 step.
- `i_glue2` (CPU latent): 4 late chunks per line, worst 0.12-0.17 s. Gapless start ~0.65-0.7 s.
- `DRAMA3_STREAM_RAMP=r[:pre]` sizes the windows for RTF r. Window [a, a+w) needs w <= (a(1-r) + w1 r + p) / r, capped at 64.
- With ramp 0.65 (`i_ramp65`, CPU latent): worst 0.05-0.07 s.
- With the quantizer on the card (`i_qdt`): **worst 0.000 s on all 8 lines** with Fish windows. The CPU latent of the long windows (78-117 ms) was the lateness.

**Proposed for `--interactive`** (Sebastian decides; main has the messages): `QWEN36_GROUPED_MM_GLUE=1`
`DRAMA3_QUANT_DEVICE=1`. With them, TTFA is 0.45-0.49 s and playback is gapless from the first sound. FIRST_WINDOW=2
stays a separate call: it gives 0.41-0.46 s first sound, but needs ~0.11 s of pre-buffer.

What is left of TTFA (~0.47 s, `i_qdt` close):
- suffix prefill backbone 0.30-0.34 s;
- tower prefill 0.043 s;
- 4 frames ~0.065 s;
- quantizer + decoder 0.030 s.

Next steps (estimates, not measured):
- A device-side expert grouping (fixed M = 96, ranks via one-hot cumsum, overflow flag read once at the end, rerun on overflow): ~-20 ms. The readback costs 44 ms with the glue, M = 96 costs ~+17 ms, the rank ops ~+6 ms.
- The GDN fused chunk op: ~1.6 ms of ~5 ms per GDN layer (synced op timing, `plops.json`), 30 layers.
- **Incremental suffix prefill while an LLM streams text** (not prototyped, needs the LLM side): prefill text chunks as they arrive. GDN state and paged K/V carry over between chunks, as the voice-prefix split does today. Today start must be a multiple of 128 (flexible chunked SDPA chunk and GDN scan chunk); chunks of 32 would need the chunked SDPA at q_chunk 32 and the GDN fused chunk at 32 (its native size). A 32-row chunk costs nearly as much as a 128-row one (weights and per-layer fixed costs dominate: 0.30 s at 77 rows). So the TTFA after the LLM's last token would be one chunk prefill (~0.25 s) + ~0.1 s ≈ 0.35 s (estimate).

### Track B: parallel voices
**Kernels** (each opt-in, bit-exact or checked; batch mode only, B = 1 still runs the solo kernels):

| lever (env) | piece, traced | check |
|---|---|---|
| `DRAMA3_GDN_FUSED_BATCH=1`: GDN decode v4 (`kernels/gdn_decode4_*.cpp`). The v3 body per (value head, sequence group): 32 x min(B, 3) cores, sequences strided. Batch conv partial sums in a [1, 2B, D] row-pair layout (sequence b = row 2b), so no two cores write one 64-byte chunk. Output rows written into a fresh L1 tensor | kernel 0.054 (v3, one row) -> 0.078 / 0.107 / 0.139 / 0.178 ms at B = 2 / 3 / 4 / 8; GDN mixer B=2 0.38 -> 0.17 ms | **bit-identical to v3 per row**: outputs, states and conv sums, B = 2 / 3 / 4 / 8 (`check_gdn_batch.json`, `check_gdn_batch2.json`) |
| `DRAMA3_MOE_FUSED_BATCH=1`: router top-k for B rows (`router_topk_b.cpp`, one core per row, the B*k id list written straight in sparse_matmul's format) + batched gather sparse_matmuls + the SwiGLU kernel + `MoECombineB` (`moe_combine_b_*.cpp`: column-broadcast weight tiles, v[b, j] at (b, 0), so one FPU multiply-accumulate per slot keeps row b of slot b*k+j; the shared expert with its sigmoid gate in the same sum) | MoE block 0.372 -> 0.280 (B=2), 0.542 -> 0.444 (B=4), 0.864 -> 0.781 ms (B=8) | vs the one-token fused path row by row: PCC min 0.999992 / 0.999992 / 0.999968 (ttnn batch path 0.99998 / 0.99998 / 0.99995, `check_moe_batch_0.json`). Column-broadcast multiplies accumulate in DST like the scalar ones (MODE 1 with explicit SFPU adds gives the same) |
| `DRAMA3_ATTN_FUSED_BATCH=1` (with ATTN_FUSED_POST + QKV_MERGED): pre / post SDPA kernels for B rows (`attn_pre_b_*`, `attn_post_b_*`; 4 / 8 cores per row), backbone and tower | backbone 17.95 -> 16.74 (B=2), 26.1 -> 24.6 (B=4); **tower 10.0 -> 8.4 ms/step** (B=2) | with all three, **B = 2 is bit-identical to B = 1** over 32 teacher-forced steps (`cbn_b7.json`); B = 3 / 4 / 8 PCC min 0.99847 / 0.99872 / 0.99894, top-1 29 / 28 / 28 of 32 (Day 5b-equivalent today: 0.9996 / 0.9984 / 0.9988) |

Backbone ms/step (check_batch_numerics, incl. stage + read; `cbn_base`, `cbn_b7`):
| B | 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| before | 27.2 | | 35.9 | 54.5 |
| after | 16.7 | 20.6 | 24.6 | 39.2 |

**Continuous batching** (`host/cbatch.py`, run via `run_check.sh <log> host.cbatch ...`). A job enters a free slot when it
arrives. Its suffix is prefilled on the one-row path in the slot's page-table row. Its GDN state is copied on the card
into the slot's row of the batch state (`kernels/row_copy.cpp`, 6 ms), and from the next step on it decodes with the
others. Freed slots are refilled. Admission policies:
- `warm-first`: no admission while a stream waits for its first window.
- `guard:<s>`: also wait until every playing stream has s of decoded audio ahead. A prefill stalls every stream for ~0.4 s.
- `greedy`.

Two things were tried and dropped:
- Adaptive windows (flush when the buffer runs low): they spiral (every flush is a card decode for all streams: RTF 0.8 -> 1.9, `cb_t2_0.3`). Removed.

**Ladder** (`/opt/ep0/port6/lad3`, final: all batch levers + glue + quantizer on the card). The workload is the 8 lines x 2 = 16 callers, one arriving every 1.0 s. That is more than the slots can serve, so queueing is reported separately. TTFA counts from admission; `rtf_decode` = a stream's decode time / its audio. All values are min / median / max over the callers:

| callers (slots) | per-stream RTF (decode) | TTFA from admission | pre-buffer for gapless playback (worst lateness) | audio s per wall s | ms/step bb + tower + host | chip / board W mean (max) |
|---|---|---|---|---|---|---|
| 1 (solo `i_qdt`) | 0.53-0.57 | 0.45-0.49 s | 0 s | 1.8 | 12.7 + 8.9 + ~1 | (Day 4b: 49-59 / 169-201) |
| **2** (`c2`, ramp 0.8:0.3, guard:0.5) | **0.65 / 0.69 / 0.72** | **0.47 / 0.50 / 0.52 s** | 0.18 / 0.20 / 0.35 s | 2.72 | 16.4 + 8.4 + 6.0 | 58.8 (99) / 199.6 (238) |
| **3** (`c3`, ramp 0.9:0.4, guard:0.5; mean 2.63 active) | 0.75 / 0.88 / 0.91 | 0.49 / 0.51 / 0.54 s | 0.28 / 0.40 / 0.65 s | 2.94 | 20.3 + 8.5 + 8.9 | 59.0 (100) / 199.7 (270) |
| 4 (`c4f`, Fish windows, warm-first; mean 3.56 active) | 0.85 / 1.03 / 1.04 | 0.50 / 0.53 / 0.62 s | 2.1-3.6 s | 3.48 | 24.7 + 8.5 + 9.0 | 65.6 (108) / 202.9 (317) |
| 4 (`c4`, ramp 0.97:0.5, guard:0.6) | 0.85-0.96, but guard kept only 1.97 active | 0.49-0.59 s | 0.36-0.50 s | 2.07 | | |
| 8 (`c8f`, mean 6.39 active) | 1.30 / 1.59 / 1.70 | 0.54 / 0.59 / 0.77 s | 6.5-12.5 s | 4.09 | 39.1 + 8.7 + 15.4 | |

Static batching with the same kernels (`lad3/s2..s8`, `run_gen.sh --batch`, the CPU-latent multi-stream codec; per-stream RTF counts the batch's prefills):

| B | audio s per wall s | per-stream RTF | TTFA | ms/step | chip / board W mean (max) |
|---|---|---|---|---|---|
| 2 | 2.37 (par/b2: 1.66) | 0.73-0.86 (1.00-1.20) | 1.0-2.6 s | 16.5 + 8.4 + 5.8 | 60.0 (130) / 207.6 (365, one 2 s sample) |
| 3 | 2.71 | 0.82-1.02 | 1.0-3.1 s | 20.3 + 8.5 + 7.8 | 64.0 / 202.3 |
| 4 | 3.12 (2.46) | 1.10-1.31 (1.40-1.59) | 1.9-3.6 s | 24.7 + 8.5 + 11.2 | 56.9 / 191.5 |
| 8 | 3.57 (2.88) | 1.79-2.37 (2.21-3.00) | 5.1-5.4 s | 39.2 + 8.7 + 20.1 | 61.2 / 189.7 |

Quality, Whisper vs written (`port6/wcheck.py` -> `sanity.json`; one seed; base `final8c` mean 0.927, min 0.837):
- s2 0.915, s4 0.920, s8 0.920, c2 0.915, c3 0.903, c4f 0.920.
- Solo i_glue2 0.915, i_qdt 0.908. No clipping anywhere.
- The low line in every glue run is specs, 0.744: Whisper hears "120 106 cores" for the line's "120 Tensix cores"; it was 0.818 on Day 4b.
- c3 also has second_opinion at 0.830.
- Name / number variants as before, but a human listen of specs is worth it.

**Headline: one p150 serves 2 callers in real time with margin, 3 just inside it, 4 at the edge.**
- 2 callers: per-stream RTF ≤ 0.72, first sound 0.47-0.52 s after admission, ~0.35 s pre-buffer for gapless playback.
- 3 callers: RTF ≤ 0.91, first sound ~0.5 s, ~0.65 s pre-buffer.
- 4 callers: median RTF 1.03, so slightly slower than real time.
- Main's estimate was 2 at 0.6-0.7 and 4 close to real time: 2 matches, 4 is ~3 % short.

**What would move it (estimates):**
- Host time per step is now the third term, 6-9 ms at B = 2-4. Most of it is the card codec decoding each stream's windows one after the other: ~30-50 ms per window, 9-16 windows per line.
- Batched codec windows (one card call for the windows that are due together, or a batch dimension in `DeviceDecoder`): ~-3 ms/step at B = 4. That alone would put 4 callers under RTF 1.
- The ramp's extra windows cost the same card time, so a smaller window overlap (32 -> 16 frames: the 32-frame decoder instead of 64) is a seam-quality question to test.
- Prefill of a new caller stalls all streams ~0.4 s. Chunked or interleaved prefill would cut the pre-buffer the guard needs.

**Files:**
- `tt/gdn_fused_decode.py` (forward_batch), `kernels/gdn_decode4_*`; `tt/moe_fused_decode.py` (RouterTopKB, MoECombineB, forward_batch), `kernels/router_topk_b.cpp`, `moe_combine_b_*`; `tt/attn_fused_decode.py` (_batch_pre / _batch_post), `kernels/attn_pre_b_*`, `attn_post_b_*`.
- `tt/quant_device.py`; `host/quant_fast.py`; `host/cbatch.py`, `kernels/row_copy.cpp`; `host/stream_codec.py` (ramp_ends, playback_check, DeviceQuantStream, CardCodec.decode_dev); `host/batch.py` (row-pair partial sums); qwen36 `moe/prefill_grouped.py` (mm glue, top-k entry) and `moe/moe.py`.
- Tests: `check_gdn_batch`, `check_moe_batch`, `check_quant_device`, `bench_grouped`, `profile_suffix_sync`, `profile_layer_ops`; `check_batch_numerics` now also prints ms/step.
- Scripts: `/opt/ep0/port6/{env,chain*,ladder,lad2,lad3,final}.sh`, `power_sum.py`, `wcheck.py`, `t_quant3.py`, `t_quant4.py`.

Full proposed batch env:
`$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1` (+ `DRAMA3_STREAM_RAMP=0.8:0.3` and `--policy guard:0.5` for 2 callers).

## Day 8 / port5 (2026-10-30): TTFA levers + fused decode kernels (all opt-in)

**Status.** Everything is opt-in and committed on `ep0/drama3` (local only, nothing pushed). The default
`run_gen.sh --fast` is still bit-identical to `day4b/out_v2` after all port5 commits (`/opt/ep0/port5/r_def2`, `r_def3`: frames
and wavs identical). Every number below comes from a run file in `/opt/ep0/port5`. The card was left idle and
`smoke.py` passed.

**All levers together** (`/opt/ep0/port5/final8c`: the 8 test lines, ref_v2, seed 0, `render_day4b.sh` with the env
below; SHA256SUMS ok, dmesg I/O errors 0 before and after, disk 69 %):
| | Day 4b final (`day4b/out_v2`) | port5 all levers (`port5/final8c`) |
|---|---|---|
| overall RTF | 0.853 | **0.57** (97.7 s of audio in 55.5 s) |
| TTFA | 2.86 s first line, 1.19-1.23 s later lines | **0.55-0.61 s on every line**, the first included |
| backbone + tower ms per frame | 23.0 + 10.5 | **12.8 + 9.1-9.3** |
| golden decode (`tests.check_paged_backbone`, trace) | 23.4 ms/step, logits PCC 0.99737, top-1 17/24 | **12.7 ms/step**, logits 0.99749, top-1 19/24; prefill layer PCC min 0.99938 (0.99719), final norm 0.99546 (0.99425), last logits 0.99778 (0.99619) (`golden_sgrid.json`) |
| Whisper vs written (8 lines, one seed) | mean 0.929 | 0.927 (cold_open 0.943 -> 1.0, server 1.0 -> 1.0, second_opinion 0.963 -> 0.868: the "cards fan spin" / "Card 2." variant the Day 4b evening run had too) |
| Whisper seeds 1-5 close / fix | 0.938 / 1.000 | 0.938 / 1.000 (`s_sgrid`; frames 133-140 / 256-278) |
| chip / board W | 49-59 / 169-201 | 50-61 / 173-223 |

```
F="DRAMA3_GDN_FUSED_DECODE=all DRAMA3_GDN_FUSED_V=3 DRAMA3_MOE_FUSED_TOPK=1 DRAMA3_MOE_FUSED_COMBINE=1 \
   DRAMA3_MOE_FUSED_SWIGLU=1 DRAMA3_ATTN_FUSED_PRE=1 DRAMA3_ATTN_FUSED_POST=1 DRAMA3_ATTN_QKV_MERGED=1"
A="DRAMA3_FIRST_WINDOW=4 DRAMA3_FIRST_WINDOW_WAIT=1 DRAMA3_PREWARM_VOICE=1 QWEN36_GDN_FUSED_PREFILL=1 QWEN36_GROUPED_SMALL_GRID=1"
EXTRA_ENV="$A $F" $D/render_day4b.sh <out dir>        # or EXTRA_ENV="$A $F" $D/run_gen.sh <log> --fast ...
```
`run_gen.sh <log> --interactive ...` = `--fast` + exactly the env above (`$A $F`; Sebastian approved the full set for live use).
Check (`port5/inter`, close / fix / model): TTFA 0.546 / 0.572 / 0.731 s (model: suffix prefill 0.52 s this run vs 0.44 s
in final8c), RTF 0.61 / 0.56 / 0.60, backbone 12.8 + tower 9.1 ms/frame, Whisper close 0.938 / fix 1.000 (`inter.gate.log`).
`--fast` re-checked bit-identical to day4b/out_v2 after the change (`port5/r_def4`).
Further opt-ins, measured, not in the set above (quality calls for Sebastian):
- `DRAMA3_FIRST_WINDOW=2` (instead of 4): TTFA 0.50-0.55 s (`port5/r11`); frames identical, audio identical from frame
  4 on, the first 4 frames differ from the 4-frame window by max 59-93 / 32768 (SNR 37-49 dB).
- `run_gen.sh ... --tower-mlp bfp8` (now with tuned one-token programs in `tt/decode_linear.py`): tower 9.2 -> 8.2
  ms/frame (`port5/r9`, RTF 0.57-0.64 on 4 lines); tower check step layer PCC min 0.99794 (bf16 0.99909), head min
  0.99658 (0.99412), head top-1 165/180 (162-166); Whisper seeds equal (`s_bfp8`). Day 4b kept BFP8 off; no
  speaker-similarity score was run.

### Track A: time to first audio (lever by lever, `port5/r0` .. `r2b`, `r10`, `tests/cmp_ttfa.py`)
Breakdown of a later line, default (r0): suffix prefill backbone 0.69-0.74 s + tower 0.043 s, first frame +0.012 s,
frames 1-10 0.30 s, the first window latent 0.07-0.11 s, its card decode 0.03 s (new timing.json fields
`prefill_backbone_s`, `prefill_tower_s`, `first_window_pushed_s`, `first_z_taken_s`, `first_card_decode_s`).
| lever (env) | before | after | check |
|---|---|---|---|
| `DRAMA3_FIRST_WINDOW=4`: one extra stream window at frame 4, Fish's 10 / 20 / 32 / 64 windows unchanged | 10 frames to the first window, 0.30 s | 4 frames, 0.10 s | frames identical; wav samples identical from frame 4 on, only the first 4 frames (~186 ms) differ, max abs diff 20-62 of 32768 (r1) |
| `DRAMA3_FIRST_WINDOW_WAIT=1`: block for the first window latent and decode it at once | latent wait 0.07-0.11 s | 0.05 s | no effect on frames / audio |
| `DRAMA3_PREWARM_VOICE=1`: the voice prefix prefill at startup | first line 2.87 s | 0.93 s (= later lines) | frames identical |
| `QWEN36_GDN_FUSED_PREFILL=1`: the upstream one-op `ttnn.transformer.chunk_gated_delta_rule` (+ explicit q / k l2 norms: the op does not normalise 4D inputs) instead of the seq adapter's glue | suffix backbone prefill 0.68-0.74 s, voice prefill 1.70 s, golden prefill 1.71 s | 0.64-0.69 s, 1.29-1.62 s, 1.59 s | golden better everywhere: layer PCC min 0.99719 -> 0.99957, final norm 0.99425 -> 0.99517, last logits 0.99619 -> 0.99725; prefix self-check cand-logit PCC 0.9986 / 0.9994 (default 0.9998), top-10 10/10; seeds equal |
| `QWEN36_GROUPED_SMALL_GRID=1` (Day 4b's grouped-MoE matmuls on the 11x10 grid below M = 256, kept off then for the suffix-vs-whole-prompt self-check alone) | suffix backbone prefill 0.64-0.69 s (with all decode kernels) | **0.34-0.44 s**; golden prefill 1.51 s | golden (with the fused prefill): prefill layer PCC min 0.99938, final norm 0.99546, last logits 0.99778, decode top-1 19/24: better than the default on every metric; prefix self-check cand-logit PCC 0.9984 / 0.9985, top-1 same, top-10 10/10; seeds 1-5 Whisper equal |
| all (final8c) | 2.86 / 1.19-1.23 s | **0.55-0.61 s** | |
What is left of TTFA (~0.58 s): suffix prefill 0.34-0.44 s (synced profile `port5/psuf.log`: GDN fused chunk ~4.9 ms
per GDN layer, the grouped MoE ~6 ms per layer, mostly its host routing readback / index build / gathers), tower
prefill 0.043 s, 4 frames ~0.07 s, the first-window latent 0.05 s (CPU: Fish's quantizer post-module, 40 matmuls +
2 transposed convs, 50 ms at 6 threads for 2-10 frames, torch.compile does not help, `t_quant2.py`), card decode 0.03 s.
Not changed: the grouped-MoE bmm configs (pitfall 18); with the small grid the suffix path no longer needs them.
Tower / next-backbone overlap: one CQ serialises them; would need sub-device partitioning (estimate, days).

### Track B: fused decode kernels (my own: `ttnn.generic_op` + kernels in `drama3/kernels/`, JIT-compiled, no C++ rebuild)
Upstream first (`git fetch` into the shared .git only, main at 7aad8aacd, 2026-10-30; no checkout): no fused
single-device GDN or MoE decode kernel beyond what our 70222e90 build has: `ttnn.transformer.fused_recurrent_gated_delta_rule`
(the GDN recurrence alone, used by qwen36's TP spec decode, wrapped in typecasts / permutes) and the fused chunk
prefill op (used above). `models/demos/deepseek_v3_b1/fused_ops` (moe, moe_routed_expert, shared_expert,
decoder_block) are fused DeepSeek decode ops for a 4x2 mesh (sigmoid gating, K = 7168, mesh reduce): reference only.
Device profile per piece (`tests/profile_ops_dev.py`: device profiler, op names via graph capture; `port5/prof_dev.json`,
`prof_all.json`): baseline GDN mixer 56 programs 369 us (its 2 weight matmuls 97 us), attention mixer 39 / 270 us,
MoE block 30 / 245 us (TopK alone 33.5 us).

| kernel (env) | piece, traced | golden decode | pipeline bb + tower | checks |
|---|---|---|---|---|
| baseline | GDN mixer 0.360, attn 0.254, MoE 0.229 ms | 23.4 ms/step | 23.0 + 10.5 | |
| GDN fused v1 (`DRAMA3_GDN_FUSED_DECODE=all`): one program, 16 cores (one per key head): conv partial sums, q/k l2 norm, gates, delta rule, gated norm; state and partials in place | GDN 0.197 | 18.2 | 17.8 + 10.5 | one layer, 8 carried steps (`tests/check_gdn_fused.py`): out PCC vs chain 0.999999-1.0, state vs an fp32 torch ref 0.999998 (chain 0.999995) |
| router top-k (`DRAMA3_MOE_FUSED_TOPK=1`): one core, data movement only, top-8 of 256 + ids as uint16 ROW_MAJOR | MoE 0.196 | 16.9 | 16.6 | same top-8 sets as ttnn.topk on 50 random + 40 real rows (order differs only on ties) |
| MoE combine (`DRAMA3_MOE_FUSED_COMBINE=1`): routed-slot sum + sigmoid-gated shared expert, 16 cores | MoE 0.171 | 15.5 | 15.6 | 40 layers PCC mean 0.99952 vs chain (`tests/check_moe_fused.py`) |
| GDN v3 (`DRAMA3_GDN_FUSED_V=3`): one core per value head (32), the 12 conv rows packed into one tile (64-byte chunks, L1 NOC copies, MEM_ZEROS), q/k partials written by the even core after a semaphore from the odd one | GDN 0.165 (v2, 16 cores packed: 0.185) | 15.0 | (next row) | o vs fp32 ref maxabs 0.003 (v1 0.012) |
| SwiGLU kernel + merged shared expert (`DRAMA3_MOE_FUSED_SWIGLU=1`, with the combine) | MoE 0.155-0.160 | 14.1 | 14.0 + 10.5 | PCC mean 0.99951 |
| attention pre-SDPA (`DRAMA3_ATTN_FUSED_PRE=1`, backbone + tower): heads packed into the [heads, 256] SDPA layout, per-row rms norm, the partial rotary as a tile swap, k / v straight into the sharded cache inputs | attn 0.180 | 13.1 | 13.2 + 9.5 | 10 layers PCC >= 0.99994 vs chain (`tests/check_attn_fused.py`); tower check top-1 164/180 (base 163) |
| attention post-SDPA (`DRAMA3_ATTN_FUSED_POST=1`): SDPA out x sigmoid(packed gate), unpacked into the o_proj row, 8 cores | attn 0.161 | 12.9 | 13.1 + 9.2 (final8) | tower check top-1 166/180 (after the dest-reuse fix, pitfall 2) |
| merged [q|gate \| k \| v] projection (`DRAMA3_ATTN_QKV_MERGED=1`, +19 MB per attention layer) | attn 0.152 | 12.6 | 12.8 + 9.1 (final8b) | outputs identical to the unmerged run |
| (no gain, not installed) one-program row norm `tt/rownorm_fused.py` | 13.9 vs 13.9 us | | | correct, closer to torch than the sharded norm |
| (slower, off) ExpertMM (`DRAMA3_MOE_EXPERT_MM=1`): both expert matmuls as an indexed matmul on 64 cores, weights over both NOCs | MoE 0.191 vs 0.156 | | | correct; its 576-byte weight tile reads run at ~2-3 GB/s per core |
| (broke, reverted) GDN v3 with the decay folded into the S reads + one-pass o norm | | | | state PCC 0.33: not understood, no speed gain |

**Measured per-layer gain:** GDN mixer 0.360 -> 0.165 ms (-54 %), MoE block 0.229 -> 0.155 ms (-32 %), attention
mixer 0.254 -> 0.152 ms (-40 %). Backbone 23.0 -> 12.8 ms/frame, tower 10.5 -> 9.1 (the attention kernels).

**Kernel pitfalls found (Blackhole, this tt-metal build):**
1. FPU scalar broadcast multiplies (`mul_tiles_bcast_scalar`) ACCUMULATE into DST; plain `mul_tiles` clears it. The
   combine kernel relies on it to sum the k slots.
2. FPU dest-reuse binaries (`*_reuse_dest_tiles<DEST_TO_SRCA>`) truncate the fp32 DST operand on its way into a source
   register (sigmoid gate: mean rel err 0.0028 vs 0.0017 rounded; tower top-1 150/180 vs 163). Use fp32 SFPU binaries
   on DST (`mul_binary_tile`, `add_binary_tile`) or pack first. The v1 / v2 GDN kernels still use dest reuse; v3 does not.
3. `sfpu_mul_bcast_col` reads more of the scalar tile than column 0 of faces 0 / 2: zero the whole tile (garbage made
   one core's outputs inf).
4. DRAM reads of 32 bytes into 32-byte aligned (not 64) L1 addresses corrupt the data: read 64-byte chunks into
   64-byte aligned slots, then NOC-copy inside L1 (16-byte alignment).
5. `EPS` is an LLK macro: do not name a kernel constant `EPS`.
6. The device profiler buffer fills during a prefill: call `ttnn.ReadDeviceProfiler` before the measured piece; the CSV
   is written at close and the trace id columns stay empty (`tests/profile_ops_dev.py` finds replays by run id).
7. Batching every read of a kernel behind one barrier made it slower (v1 0.197 -> 0.213, v3 0.175 -> 0.186): stream the
   inputs the compute needs first.
8. Each core pulling a full shared [1, 2048] row as 64 tiles from the L1 banks congests the NOC (more cores were
   slower); read row-0 chunks and build the tiles locally.

**What it would take to go further (estimates, not measured):**
- MoE (~6.2 ms/frame): the two sparse expert matmuls (45 + 32 us per layer, ~240 GB/s) are the largest backbone
  item; a fused expert kernel near ~400 GB/s: ~-1.2 ms/frame (days; my ExpertMM shows the weight stream is the hard part).
- GDN fused op ~35 us of the 165 us mixer (in-projection 72 + o_proj 24 us at ~350 GB/s): ~-0.3 ms/frame.
- Norms: ~145 one-token norms per frame at 7.4 us (3 programs); keeping the residual width-sharded so I2S / S2I drop
  out: ~-0.3 ms/frame.
- Tower (9.1 ms, 13 layer-steps per frame) is bf16-weight bound: the BFP8 MLP above gives -1.0 ms.
- TTFA: the grouped-MoE suffix prefill reads the routing to the host every layer (40 syncs); a device-side grouping
  (or the indexed decode path at small M) and the quantizer post-module on the card (-45 ms) are the next steps.
- Weight floor: the backbone reads ~2.3 GB per frame (~4.5 ms at DRAM speed); 12.8 ms now.

**Files:** kernels `drama3/kernels/{gdn_decode,gdn_decode2,gdn_decode3,router_topk,moe_combine,swiglu,expert_mm,attn_pre,attn_post,rownorm}_*.cpp`;
hosts `tt/gdn_fused_decode.py`, `tt/moe_fused_decode.py`, `tt/attn_fused_decode.py`, `tt/rownorm_fused.py`; installs in
`tt/backbone_paged.py` (GDN, MoE, attention) and `tt/tower_device.py` (attention); checks `tests/check_gdn_fused.py`,
`check_router_topk.py`, `check_moe_fused.py`, `check_attn_fused.py`, `check_rownorm.py`, `profile_ops_dev.py`,
`cmp_ttfa.py`; TTFA code in `host/generate_fast.py`, `host/stream_codec.py`; the fused prefill switch in
`models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`; tuned BFP8 tower entries in
`tt/decode_linear.py`. Run scripts: `/opt/ep0/port5/{go,golden,seeds}.sh`, chains `chainB*.sh`, `chainC*.sh`.
Commits: 96d053313, c424d098e, f01789476, a82e99625, c8641814e, 604a925b2, 0ee161794, c6fa9656d, a29dbab4b, + this.

## Day 7 / state for the morning (ep0-port4, 2026-10-30 ~05:00): read this first

**Frozen.** Everything is committed on `ep0/drama3`: the code, the scripts, this file, and the Day 6/7 scorers. Nothing
was pushed. The card was left idle; the smoke test at the end of Day 7 passed. The whole Day 3 → Day 6 scoreboard is
in `/opt/ep0/day7/SCOREBOARD.md` + `scoreboard.json`. `tests/day7_scoreboard.py` generates both from the run files;
it makes no card or Whisper runs. Each number cites its source file, and "missing" marks values that were never
measured.

Headline numbers (from the scoreboard):
| | Day 3 | Day 4 (ref_v2) | Day 4b final (ref_v2) | Day 6 narration (38 lines) |
|---|---|---|---|---|
| overall RTF | 14.7 (ref_long) | 1.91 | **0.853** | **0.961** |
| TTFA first / later lines | 33.7 / 17.4-20.9 s | 3.05 / 3.05-3.80 s | 2.86 / 1.19-1.23 s | 3.72 / 1.13-1.56 s |
| ms per frame backbone + tower | 273 + 271 | 45.6 + 20.5 | 23.0 + 10.5 | 23.0 + 10.8 |
| codec | CPU, whole clip after generation | CPU stream, 51.7 ms/frame | card, 1.6 ms/frame card time | card |
| card DRAM | 25.96 GB (backbone only) | 27.72 GB (HANDOFF) | 28.29 GB | 28.29 GB |
| Whisper mean vs written (small) | 0.947 | 0.932 | 0.929 | 0.941 |
Batch (8 test lines; audio s per wall s): B1 1.17, B2 1.68, B4 2.31, B8 2.70.

**Default (no flags beyond `--fast`):** `run_gen.sh <log> --fast ...` with the Day 4b pipeline: voice prefix cache,
device codebook sampling, traced backbone + fused tower trace, card codec at 3 window sizes, indexed top-k MoE
decode, BFP4 gate/up experts, seed 0. After all the Day 5/5b/6/7 patches it is bit-identical to `day4b/out_v2`
(`/opt/ep0/day5/gate_base`, `/opt/ep0/day5b/final_default`).
**Opt-in:** `--batch --batch-size B` (Day 5b, static batching; B>1 is not per-row identical to solo, but the
quality is equal), `QWEN36_EXPERT_GU_BFP8_LAYERS=...` (Day 5, BFP8 gate/up on chosen layers: +2.1 GB for
8 layers, no measurable gain), `DRAMA3_PREFIX_CACHE=0` (the Day 4 whole-prompt path), `DRAMA3_TOWER_DEVICE_SAMPLING=0`,
`DRAMA3_SAMPLER_CHECK=1`. Never on: `QWEN36_GROUPED_BMM_CONFIG=1` (non-deterministic, pitfall 18).

**Narration retries (Day 7 Task 1):** port3's `/opt/ep0/day6/retry2.sh` rendered them, 24 takes in
`/opt/ep0/day6/narration_retry`: s04-05, s05-02, and s02-01 / s02-02 / s10-02 as "orig" and as a "mote" variant
(moat → mote, cloud bills → cloud bill's), seeds 1-3, default pipeline, sequential, SHA256SUMS OK, 0 I/O errors; power log gap of ~20 s from 04:37:38 (a duplicate Day 7 run stopped the sampler, restarted).
They were scored twice: port3's `tests/day6_variants.py` (`picks.json`) and Day 7's `tests/day7_retry_pick.py`
(`picks_day7.json`). The Day 7 scorer also includes the canonical seed-0 takes and the `moat_heard` /
`cloud_bills_heard` flags (true = heard by both models). The models are faster-whisper small + base, both
multilingual with language=en; the server has no .en models and nothing was downloaded. Result: neither model hears
"moat" in any of the 15 takes per moat line, mote spelling included; they hear "mode". "cloud bills" is always
"cloud builds". Recommendation: s04-05 → `narration_retry/s04-05_orig_s1.wav` (both models "felt"; on the canonical
take base hears "felled"); keep the canonical take for s05-02, s02-01, s02-02 and s10-02. Moat → mode is most likely
a Whisper effect (the close gate always showed it). A human listen decides. `/opt/ep0/day6/narration` is unchanged.

**Open levers (estimates, not measured):**
1. Batched codec: the static batch's windows end on the same frames, so decode them in one card call instead of
   B calls one after the other. 15.1 of 19.6 ms host per step at B=8, so ~-10 ms/step at B=8.
2. Continuous batching: start a stream's decode right after its own suffix prefill and refill freed slots. TTFA at
   B>1 would go back to ~1.2 s (today 2.1-9.8 s), and finished streams would stop wasting their slots.
3. MoE at B=8: 0.79 ms/layer (vs 0.22 at B=1) because each of the B*8 slots reads its own expert. A dedupe of
   shared experts needs dynamic shapes or a sort+segment kernel.
4. Fused kernels for the backbone decode: ~100 ops per layer at ~5-10 us dispatch; the weights need only ~4.5 ms per
   frame at DRAM speed against 22.5 ms of device time. Fused GDN-decode and MoE-decode kernels are the only big step
   left for B=1 (days of kernel work).
5. Smaller: TTFA under 1 s (suffix GDN chunk prefill, Fish's 10-frame first window), vectorised host draws at B>1.

**Reproduce the episode numbers** (on the server, D=`/opt/ep0/tt-metal/models/demos/blackhole/drama3`):
- 8 test lines, Day 4b final: `nohup $D/render_day4b.sh /opt/ep0/<new dir> > <log> &`, then
  `/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_day4 /opt/ep0/<new dir>"`.
  It should be bit-identical to `/opt/ep0/day4b/out_v2` (frames + wavs).
- Episode narration (38 lines): `nohup $D/render_day6.sh /opt/ep0/<new dir> > <log> &` (writes timing.json,
  sanity.json, SHA256SUMS). Batch-8 set: `$D/render_day6_batch.sh 8`.
- Batch ladder: `run_gen.sh <log> --batch --batch-size B --lines all --ref /opt/ep0/voice/ref_v2.wav --out <dir>`
  for B = 2, 4, 8 (`batch_summary.json` per dir).
- Retries: `/opt/ep0/day6/retry2.sh` (variants in `narration_retry/variants.json`), then `/opt/ep0/day7/pick_day7.sh`.
- Scoreboard: `cd /opt/ep0/tt-metal && python3 models/demos/blackhole/drama3/tests/day7_scoreboard.py /opt/ep0/day7`.
- Card health: `/opt/ep0/run.sh "python /opt/ep0/smoke.py"` (SMOKE_OK, PCC 0.9999).
Not run overnight: the Windows-side scorer (WER + speaker similarity) on the Day 6 batch-8 set and on the Day 7 retries.

## Days 5 / 5b / 6 (ep0-port3, 2026-10-30): status first

Day 5 done: keep BFP4 (BFP8 on the 8 most sensitive layers: +0.0003-0.0007 PCC, same top-1, no Whisper gain, +2.1 GB).
Day 5b done: batch mode works (opt-in `run_gen.sh --batch --batch-size B`), 2 voices at real time in steady state,
2.7 audio s per s at 8 voices (2.3x solo). Day 6 done: 38 narration lines in `/opt/ep0/day6/narration` (sequential,
RTF 0.96) + a batch-8 set in `/opt/ep0/day6/narration_b8` (1.96 audio s per s). Commits 7910d6bd, d892ed8b, 4597fa7c, this.

## Day 5: expert precision BFP4 vs BFP8 (2026-10-30, ep0-port3) — decision: keep BFP4

**BFP8 gate/up for all 40 layers does not fit**: a BFP8 gate or up file is 285 MB per layer vs 151 MB BFP4
(+134 MB per projection per layer), i.e. +10.7 GB for all layers vs 5.9 GB free on the card (28.29 / 34.18 GB in
the full pipeline). (The "~4.8 GB" estimate in the Day 4b next-agent notes was wrong.) So two measurements:

1. **Weight format alone, CPU FP32 math** (`reference/cpu_layer_ref.py --rows close.rows_teacher24.pt --quant
   qwen36|bfp8`, `/opt/ep0/day5/sim_qwen36`, `sim_bfp8`; `tests/compare_sims.py` vs `/opt/ep0/day3/golden_voice`,
   error = 1 - PCC of the RMS-normalised residual without channel 1269, `/opt/ep0/day5/sims_compare.json`):
   | experts gate/up (other weights as on the card) | final norm PCC | cand logits PCC (25 teacher rows) | top-1 | top-5 overlap |
   |---|---|---|---|---|
   | BFP4 (today) | 0.99691 | 0.99848 | 22/25 | 4.44 |
   | BFP8, all 40 layers | 0.99871 | 0.99930 | 24/25 | 4.68 |
   The card today (golden check) has decode logits 0.9974 / top-1 17/24: the device math (LoFi decode,
   bf16 activations) costs more than the BFP4 weights. Most sensitive layers by the per-layer step error
   (BFP4 - BFP8): 36, 37, 0, 30, 32, 28, 27, 33, 29, 31, 25, 23, 26, 2, 15, 7 (the late layers).
2. **BFP8 gate/up on the 8 most sensitive layers on the card** (`QWEN36_EXPERT_GU_BFP8_LAYERS=0,27,28,30,32,33,36,37`;
   cache files written by `tests/convert_gu_bfp8.py` via `/opt/ep0/day5/convert_gu.sh`: 16 files, 4.6 GB, sha256 as
   written == from disk after drop caches, `/opt/ep0/day5/cache_bfp8.sha256`, dmesg I/O errors 0, disk 69 %):
   | variant | golden prefill layer PCC min / final norm / last logits | decode logits PCC / top-1 | ms/frame bb + tower | card DRAM | Whisper close / fix (seeds 0,1,2) | frames close / fix |
   |---|---|---|---|---|---|---|
   | BFP4 (default) | 0.99719 / 0.99425 / 0.99619 | 0.99737 / 17/24 | 23.0 + 10.5 | 28.29 GB | 0.938 / 1.000 (seeds 1-5, Day 4b) | 131-142 / 264-277 |
   | BFP8 x 8 layers | 0.99755 / 0.99493 / 0.99643 | 0.99764 / 17/24 | 23.1 + 10.5 | 30.43 GB | 0.938 / 0.982 (fix_s0: "PowerSupply's") | 131, 134, 151 / 276, 274, 281 |
   (`/opt/ep0/day5/golden_base.json`, `golden_bfp8_8.json`, `gate_bfp8_8/`, `whisper_bfp8_8.log`.)

**Decision: keep BFP4.** BFP8 on the 8 most sensitive layers moves the golden numbers by +0.0003..0.0007 PCC with
the same top-1, no Whisper gain, costs 2.1 GB of card DRAM (which batching wants) and buys nothing measurable in
speed either way. Full BFP8 would need the card codec or the tower off the card. The 8 test lines with the BFP8
variant are in `/opt/ep0/day5/out_bfp8_8/` (same format as day4b/out_v2) for the Windows similarity scorer.
Default pipeline regression after the Day 5/5b patches: golden check identical to `golden_qs.json`, close + fix
frames and wavs bit-identical to `day4b/out_v2` (`/opt/ep0/day5/gate_base`).

## Day 5b: parallel voices, batch > 1 (2026-10-30, ep0-port3) — works, opt-in

`run_gen.sh <log> --batch --batch-size B --lines ... [--lines-file ...] --ref ... --out <dir>` (`host/batch.py`;
the default `--fast` pipeline is untouched: after all Day 5b patches close + fix frames AND wavs are bit-identical
to day4b/out_v2, `/opt/ep0/day5b/final_default`). Static batching: up to B lines start together (sorted by
length), a finished stream keeps its slot until the batch ends.

How it works: `DRAMA3_KV_STREAMS=B` makes the backbone / tower K/V a pool of B sequences; stream b's page-table
row = the shared voice-prefix blocks (prefilled once, read only) + its own blocks (row 0 = the identity table, so
stream 0 = solo layout). Each line's suffix is prefilled on the one-row path in its page-table row, its GDN state
(recurrent [1,32,128,128] + the 3 conv partial-sum rows) is read to the host and all B rows are uploaded into the
batched GDN buffers in one go (34 ms at B=2). Then one backbone trace + one tower trace per step for B rows: the
attention / GDN mixers take [1, B, H] as B one-token sequences (`rows_are_batch` attribute; projections, conv,
norms, gates on one tile, rope / K/V / sdpa-decode / the recurrence per sequence), the MoE runs a batched indexed
top-k decode (`QWEN36_MOE_DECODE_INDEXED_BATCH=1`: the B*8 routed ids as one gather list, a block mask times the
routing weights keeps each row's own 8 slots), the one-tile norms / tuned linears take B rows
(`decode_norm.ROWS`, `decode_linear.ROWS`), the device codebook sampler runs per row. Host: each stream's main
token from its own generator in the solo RNG order (noise for the next frame drawn while the backbone trace
runs), the card codec per stream via a multi-stream latent process (`MultiCardStreams`).

Gates: **B=1 batch mode = solo, frames bit-identical** (close, fix; `/opt/ep0/day5b/b1`). B>1: the B rows are
bit-equal to each other, but not to solo: teacher-forced backbone logits (32 steps, B identical streams) vs the
B=1 math PCC min 0.9988 / 0.9995 / 0.9990 at B=2/4/8, top-1 31/30/31 of 32 (`tests/check_batch_numerics.py`,
`/opt/ep0/day5b/check_batch_numerics.json`), so the codes diverge after frame 0-1 (Fish's rank-noise sampler,
pitfall 8). Quality equal: close at B=2 (seeds 0,1) and B=4 (seeds 1-4) Whisper 0.938 (= solo), frames 136-145
(solo seeds 131-146); the 8 test lines at B=2/4/8 Whisper as solo within name/number variants, no clipping.

Device time per step, traced (`tests/profile_batch.py`, `/opt/ep0/day5b/profile_batch2.json`, `profile_batch3.json`):
| B | backbone | tower | per layer: GDN mixer / attention mixer / MoE | MoE dense-union path instead |
|---|---|---|---|---|
| 1 | 22.2 ms | 9.3 ms | 0.37 / 0.25 / 0.22 ms | 0.22 |
| 2 | 26.5 | 9.8 | 0.38 / 0.28 / 0.30 | 0.99 |
| 4 | 35.3 | 10.5 | 0.45 / 0.35 / 0.47 | 1.12 |
| 8 | 53.5 | 11.8 | 0.58 / 0.43 / 0.79 | 1.45 |
(The first version with the mixers on [B, 1, H] was 22 / 35 / 57 / 101 ms: one tile per row.)

Measured, the 8 test lines (ref_v2, seed 0; `/opt/ep0/day5b/m2`, `m4`, `m8`, each with `batch_summary.json`,
`sanity.json`; per-stream RTF and TTFA count from the batch start, i.e. they include the sequential suffix
prefills of all B streams):
| streams | audio s per wall s | RTF per stream | TTFA | ms per step bb + tower + host | chip / board W | card DRAM |
|---|---|---|---|---|---|---|
| 1 (solo, day4b/out_v2) | 1.17 | 0.82-0.96 | 1.2 s (first 2.9) | 23.0 + 10.5 + ~1 | 49-59 / 169-201 | 28.29 GB |
| 2 | 1.68 | 1.05-1.22 | 2.1-3.9 s | 27.2 + 10.0 + 5.6 | 44-79 / 156-245 | 28.53 |
| 4 | 2.31 | 1.49-1.77 | 3.8-5.8 s | 36.1 + 10.7 + 9.9 | 44-94 / 152-222 | 28.81 |
| 8 | 2.70 | 2.45-3.37 | 9.4-9.8 s | 54.3 + 12.1 + 19.6 | 46-121 / 176-242 | 29.39 |
**Headline: one p150 serves 2 voices at real time each in steady state (42.8 ms per step < 46.4 ms), 1.05-1.2
including the batch's prefills; the most audio per second is 2.7 s/s at 8 voices (2.3x one voice).** Of the
19.6 ms host time per step at B=8, 15.1 ms is the card codec decoding the 8 streams' windows one after the other.

Open levers (estimates): batch the codec windows across streams (they end on the same frames in a static batch:
-10 ms/step at B=8), continuous batching (start decoding a stream right after its prefill: TTFA back to ~1.2 s),
the MoE at B>1 (0.8 ms/layer at B=8: B*8 slots each read their expert; a dedupe would need dynamic shapes),
vectorised host draws.

## Day 6: the episode narration on the card (2026-10-30, ep0-port3)

`/opt/ep0/day6/narration/<id>.wav` (canonical set): the 38 lines of `/opt/ep0/voice/card_lines.json` (text =
`spoken`), ref_v2, BFP4, sequential `--fast` pipeline (`render_day6.sh`), seed 0; plus timing.json, sanity.json
(Whisper vs `written`), picks.json, SHA256SUMS (sums of the bytes as written, verified from disk after sync + drop
caches). **202.2 s of audio in 194.3 s compute = RTF 0.96 overall** (lines 0.85-1.74; short lines pay prefill +
TTFA, the 1.74 is the first line with the voice prefill), TTFA 1.1-3.7 s, backbone 23.0 + tower 10.8 ms per frame,
chip 43-79 W, board up to 214 W, card DRAM 28.29 GB, dmesg I/O errors 0, mean Whisper match 0.941.
Lines < 0.85 re-rendered with seeds 1-3 (`/opt/ep0/day6/retry`, `tests/day6_pick.py`): s03-01 0.471 -> 0.800 with
seed 2 (kept; the seed-0 take is `retry/s03-01_s0.*`). Still < 0.85, all Whisper / name / placeholder artefacts,
kept at seed 0: s01-01 0.833 ("$1400 AI card"), s05-02 0.833 ("Card 2."), s06-11 0.778 (the "[it makes no
difference / the difference]" placeholder: the model reads "[...]" as a tag and says "the difference"), s08-03
0.848 ("[n] days" -> "days", "10 store and"); s03-01 0.800 ("10-storrent").
Batch-mode alternative (`render_day6_batch.sh 8`, `/opt/ep0/day6/narration_b8`, same files + batch_summary.json):
199.7 s of audio in 102.0 s wall = 1.96 audio s per s (1.9x the sequential set), per-stream RTF 2.7-5.9 and TTFA
5.5-13 s (static batches of 8), chip 43-75 W, board 160-250 W (one 2 s sample at 341 W), card DRAM 29.39 GB, mean Whisper 0.932, < 0.85: s01-01 0.800,
s03-01 0.800, s05-01 0.839, s05-02 0.833, s06-11 0.778, s08-03 0.812 (not re-rendered).

## FINAL STATE of Day 4b (read this first)

**Faster than real time on one p150.** Final 8 lines with ref_v2, seed 0, in `/opt/ep0/day4b/out_v2` (timing.json,
sanity.json, SHA256SUMS verified after sync + drop caches, dmesg I/O errors 0; rendered 2026-10-30 02:50 CET with
`render_day4b.sh`, defaults of `run_gen.sh --fast` at commit 81541859 and later): **overall RTF 0.853** (96.9 s of
audio in 82.6 s), **every line below 1.0**: 0.82-0.91, the first line (price) 0.962 including the 1.70 s voice
prefill. **TTFA 1.19-1.23 s** on every line after the first, 2.86 s on the first. Backbone 23.0 + tower 10.5 ms per
frame (real time is 46.4 ms), card codec 0.28-0.46 s per line, codec tail 0.14-0.20 s. Card DRAM 28.29 / 34.18 GB.
Chip 49-59 W, board 169-201 W. Whisper match vs the written form equal to Day 4 on 7 lines (specs 0.837 -> 0.818:
"D Flash 2"); misses only names / numbers as Day 4; no clipping; frames within -3.6 / +3.8 % of Day 4 out_v2.
Multi-seed check (seeds 1-5, close + fix): Whisper mean match identical to the Day 4 math, frame counts in the same
range. Not run: the Windows-side scorer (WER + speaker similarity, `ExitTheMoat-github/.../eval/score.py`, needs
faster_whisper + speechbrain on Windows and appends to the shared results file).

Day 4 -> now: backbone 45.6 -> 23.0 ms/frame, tower 20.5 -> 10.5, CPU codec (51 ms/frame + 2.5-3.6 s tail) -> card
codec, TTFA 3.05 -> 1.2 s, RTF 1.91 -> 0.85. Lever tables below (evening: levers 1-2, overnight: 3-19); every
lever was committed only after the golden check (`tests.check_paged_backbone`) and the close/fix gate passed.
Device time per frame now (`tests/profile_frame.py`): backbone trace 22.5 ms + fused tower trace 9.2 ms = 31.7 ms;
host 1.8 ms. What is left: ~100 ops per backbone layer at ~5-10 us dispatch each (the backbone reads ~2.3 GB of
weights per frame, ~4.5 ms at DRAM speed); only fused custom kernels move that further.

### Next agent: Day 5 -> 5b -> 6

The card is free (no job running); start from this state (commit c77d8d18 or later, defaults of `run_gen.sh --fast`).
1. **Day 5: expert precision BFP4 vs BFP8.** The routed experts are gate/up BFP4, down BFP8 (Day 2 tensor cache).
   Build a BFP8 gate/up cache (disk precautions: throttled writes, sha256 after sync + drop caches, keep < 85 %),
   then compare on the golden check (`tests.check_paged_backbone`), the multi-seed Whisper check
   (`tests/gate_seeds.py`) and the Windows scorer (WER + similarity), plus ms/frame (the indexed MoE decode reads
   2x the gate_up bytes: expect +1-2 ms/frame) and card DRAM (28.3 / 34.2 GB used now, BFP8 gate_up adds ~4.8 GB:
   check it fits next to the card codec).
2. **Day 5b: batch > 1 / parallel voices sharing the prefix cache.** The decode trace, tower, sampler, and codec are
   written for one sequence (paged K/V with an identity page table, one GDN state set). Parallel lines need
   per-sequence page tables + GDN state copies (the prefix K/V blocks can be shared read-only; `snapshot_gdn` /
   `restore_gdn` already copy the GDN prefix state), the indexed MoE decode is batch-1 only (`decode_forward_indexed`;
   the dense path handles B > 1), and the card codec runs per stream between frames.
3. **Day 6: render `/opt/ep0/voice/card_lines.json` (38 lines)** with the `--fast` defaults and ref_v2, like
   `render_day4b.sh` (it reads `lines.yaml` today: add a lines-file option to `host/generate_fast.py`), then
   summarize (`tests.summarize_day4`), SHA256SUMS cold, and the Windows scorer.

Day 4b output dirs: `out_v2` = final (all levers), `out_v2_first` = the evening set (the requested `out_v2_l2`
name was not used; same content role), `out_v2_cp1..cp3` = checkpoints.

Previous (evening) result, superseded: Day 4b goals were TTFA (voice prefix cache) and an honest real-time verdict.
8 lines with ref_v2 then (now in `/opt/ep0/day4b/out_v2_first`):
**overall RTF 1.71** (Day 4: 1.91; per line 1.61-2.01), **TTFA 1.56-1.66 s for every line after the first**
(Day 4: 3.05-3.8 s), first line 3.60 s (includes the 1.97 s voice prefill, once per session/voice), backbone
45.4 ms + tower 17.1 ms per frame (Day 4: 45.6 + 20.5), card DRAM 27.86 / 34.18 GB, chip 48.7-54.6 W, board
171-190 W. Whisper match vs the written form equal to Day 4 on 6 lines, cold_open 0.943 -> 1.0, second_opinion
0.963 -> 0.868 ("the cards fan spin" for "the card's fans spin", "Card 2." for "Card two:"); misses otherwise only
Tenstorrent / Qwen / QSFP-DD / numbers as Day 4; no clipping; frames within
±4.2 % of Day 4 out_v2 per line.

| lever (close + fix, ref_v2, seed 0) | backbone / tower ms | TTFA first / later | RTF close / fix | gate |
|---|---|---|---|---|
| Day 4 baseline (`day4/out_v2`) | 45.6 / 20.3 | 3.8 / 3.05 s | 2.51 / 1.90 | - |
| 1 voice prefix cache, split 640 (`day4b/gate1`) | 45.3 / 19.3-20.2 | 3.94 / 1.95 s | 2.41 / 1.80 | cand-logit PCC 0.9998 vs whole prompt, same top-10; Whisper OK |
| 1b grouped-MoE M buckets 32/64/96 (`gate2c`) | 45.5 / 20 | 3.53 / 1.60 s | - / 1.75 | golden check = baseline (final norm 0.994, decode logits 0.997, top-1 18/24) |
| 2 codebook sampler on the card, tower = 1 trace (`gate3b`) | 45.6 / 16.8 | 3.57 / 1.60 s | 2.28 / 1.71 | codes bit-identical to the host sampler (138/138, 263/263 frames); host re-draw check 0 / 3,654 |
| (rejected) bmm program configs for the grouped MoE | - | voice 1.59 s, suffix 0.60 s | - | BROKE the prefill in context (final norm PCC 0.84, a NaN layer) -> off |
| (rejected) codec tail with 14 threads | - | - | - | tail 3.41 -> 3.58 s (fix), 5.8 s (close): no gain |

Overnight levers (after the out_v2 render above; each committed when it passed; defaults in `run_gen.sh --fast`):

| lever | backbone / tower ms | TTFA later | RTF close / fix | gate |
|---|---|---|---|---|
| 3 indexed top-k MoE decode (`QWEN36_MOE_DECODE_INDEXED=1`, `MOE_INDEXED=0` to disable): sparse_matmul gather mode on the router's top-8 ids, compact [8, M, N] outputs, so matmuls and glue skip the other 248 expert slots (`gate5`) | 37.4 / 17.0 | 1.54 s | 2.15 / 1.54 | golden: decode trace 44.6 -> 36.7 ms/step, final norm 0.9926, logits 0.9973, top-1 18/24 (= baseline); Whisper OK, frames 137 / 276 |
| 4 split GDN conv state in decode (`DRAMA3_GDN_SPLIT_CONV`, default 1): 3 [1,1,D] rows, qwen36's in-place split path, no slice/concat/to_layout per tap; prefill keeps the fused buffer and copies it into the rows at its end (`sync_split_conv`) (`gate6`) | 35.3 / 17.0 | 1.51 s | 2.09 / 1.48 | golden identical to lever 3 (decode 36.7 -> 34.7 ms/step); frames bit-identical to gate5 |
| 5 one-token RMSNorms width-sharded on 8 cores (`tt/decode_norm.py`, `DRAMA3_SHARDED_DECODE_NORM`, default 1): 36.5 -> 13.9 us per norm incl. the conversions; backbone 81 + tower 65 norms per frame (`gate7`) | 33.2 / 16.0 | 1.49 s | 2.23 / 1.46 | golden: decode 34.7 -> 32.9 ms/step, final norm 0.9928, logits 0.9974, top-1 18/24; tower check: step layer PCC min 0.9991, head min 0.9966 (base 0.9990 / 0.9949); Whisper OK, frames 131 / 258 |
| 6 tuned one-token matmul programs (`tt/decode_linear.py`, `DRAMA3_DECODE_LINEAR`, default 1; table from `tests/tune_decode_linear.py`): 1D mcast configs per weight signature, e.g. router 51 -> 13 us, o_proj 84 -> 33 us, shared-expert gate 49 -> 8 us, tower w2 122 -> 70 us (`gate8`) | 27.1 / 14.0 | 1.39 s | 1.88 / 1.26 | golden: decode 32.9 -> 26.2 ms/step, final norm 0.9930, logits 0.9974, top-1 18/24; tower check step layer min 0.9985, head min 0.9965, top-1 166/180; Whisper OK, frames 136 / 276 |

After lever 6 the generation loop (41 ms/frame) is faster than real time (46.4 ms); the CPU codec (13.2 s for
276 frames = 48 ms/frame in Fish's windows) is now the limit, plus its 3.5-3.7 s tail per line.

| lever | backbone / tower ms | TTFA later | RTF close / fix | gate |
|---|---|---|---|---|
| 7 codec decoder on the card (`tt/codec_device.py`, `host/stream_codec.py` CardCodec; `DRAMA3_CARD_CODEC`, default 1): the codec process computes only the window latents (quantizer, 1.2 ms/frame CPU), the generation loop decodes each window on the card at one fixed length (96 frames, zero latent rows appended; the decoder is causal) between frames: 97 ms per window, fp32 activations, conv inputs typecast to bf16 first (`gate11`) | 26.3 / 12.6 | 1.29 s | 1.38 (first line, incl. 1.96 s voice prefill) / **0.98** | codes identical to gate8; audio vs the CPU codec: SNR 43.7-44.3 dB, PCC 0.99998, rms -0.03 dB, peak 0.93, no clipping; codec tail 3.7 -> 0.19 s; Whisper OK |
| 8 router: top-k on the raw scores, softmax over the 8 (`QWEN36_ROUTER_TOPK_FIRST`, default 1 in `run_gen.sh --fast`) (`gate12`) | 25.7 / 12.5 | 1.31 s | 1.36 / 0.98 | golden: decode 26.2 -> 25.5 ms/step, final norm 0.9928, logits 0.9974, top-1 18/24; Whisper OK, frames 143 / 275 |
| (kept off) routing weights before down_proj (`QWEN36_MOE_DECODE_PRESCALE=1`) | - | - | - | golden passes (0.9932 / 0.9975 / 17 of 24), only 25.5 -> 25.4 ms/step |
| (kept off) tower MLP BFP8 (`--tower-mlp bfp8`) | - | - | - | tower check: step layer min 0.9982, head min 0.9952, top-1 160/180 (bf16: 0.9985 / 0.9965 / 166); est. -1.5 ms/frame |
| (kept off) codec decoder as one trace (`DRAMA3_CODEC_TRACE=1`) | - | - | - | audio bit-identical, but no gain: the 90-97 ms per window are device time |
| 9 host work overlapped with the backbone trace (`DRAMA3_HOST_OVERLAP`, default 1): `bb.decode_start` / `tower.prepare_frame` (all of the frame's noise draws in the same RNG order, the shared input) / `bb.decode_finish`; the 5 host embedding rows in one upload; Fish's main-token sampler on the pre-drawn noise with a top-64 shortcut (`sample_with_noise`) (`gate14`) | 25.6 / 11.0 | 1.27 s | 1.32 / **0.925** | frames AND audio bit-identical to gate12 |
| 10 partial RoPE as one matmul in one-token decode (`DRAMA3_ROPE_MATMUL`, default 1; qwen36 `register_partial_rope`): full-width cos / sin ([cos\|1], [sin\|0]) and x * C + (x @ P) * S with a +-1 permutation P (HiFi4), 8 ops instead of ~20 per attention layer (10 backbone + 13 tower) (`gate15`) | 25.4 / 10.6 | 1.26 s | 1.31 / 0.919 | golden bit-identical to lever 8; frames and audio bit-identical to gate14 |
| 11 GDN decode conv as running partial sums (`DRAMA3_GDN_CONV_PARTIALS`, default 1; `ConvPartialSums` in ttnn_gated_deltanet): y = Q0 + w3 x, Q_j = Q_{j+1} + w_{2-j} x, Q2 = w0 x (+ bias), via addcmul with output_tensor; prefill end converts the last 3 inputs into the partials (`conv_partials_from_states`) (`gate16`) | 24.6 / 10.6 | 1.27 s | 1.31 / 0.93 | golden: decode 25.3 -> 24.5 ms/step, final norm 0.9927, logits 0.9973, top-1 18/24; Whisper OK, frames 142 / 276 |
| 12 lean GDN recurrent decode step (`QWEN36_GDN_LEAN_DECODE`, default 1 in `run_gen.sh --fast`; `recurrent_gated_delta_rule_decode_lean_ttnn`): same ops on the same values, rows kept as [B, H, 1, X] (no 3D <-> 4D reshapes) (`gate17`) | 24.0 / 10.6 | 1.26 s | 1.27 / **0.89** | golden bit-identical to lever 11; frames and audio bit-identical to gate16 |
| (rejected) codec windows at 3 fixed sizes (`DRAMA3_CODEC_WINDOWS=32,64,96`) | - | - | - | the second decoder runs out of L1_SMALL (conv1d keeps per-program sliding-window buffers there); default "96" |
| (kept off) grouped-MoE bmm configs, retested 3x: **non-deterministic**: golden passes twice (prefill final norm 0.9937 / 0.9939) and fails once (0.956, a NaN layer); check_prefix_cache suffix row PCC 0.83-0.93. Voice prefill would be 1.95 -> 1.59 s. Stays off until the race is found | - | - | - | - |
| (kept off) tower MLP BFP8, retested in the pipeline: tower 10.6 -> 10.9 ms (no tuned program for the BFP8 shapes) | - | - | - | - |
| (kept off) codec bf16 activations with fp32 snakes (`DRAMA3_CODEC_DTYPE=bfloat16 DRAMA3_CODEC_SNAKE_FP32=1`): 91.7 ms per window (no gain) and +2 % level (the bias is in the bf16 conv/linear outputs, not the snakes) | - | - | - | - |
| 13 tuned program configs for silu-activated one-token linears too (the shared expert's w1, 40 per frame): the silu moves into the program config (`fused_activation`) (`gate20`) | 23.3 / 10.7 | 1.27 s | 1.27 / 0.88 | golden: decode 24.4 -> 23.7 ms/step, final norm 0.9926, logits 0.9973, top-1 17/24; Whisper OK, frames 141 / 267 |
| (no change) MoE decode K blocks re-swept for the indexed path (`tests/sweep_moe_blocks.py`): 32 / 8 stays best (22.55 ms device per decode step) | - | - | - | - |
| 14 card codec snakes fused (`DRAMA3_CODEC_SNAKE_FUSED`, default 1): multiply with sin + square post-activations, then addcmul: 2 passes instead of 5 (snakes were 54 of 111 ms per window, `tests/profile_codec_stages.py`); 96.7 -> 73.2 ms per window (`gate21`) | 23.3 / 10.7 | 1.27 s | 1.24 / **0.86** | codes identical; audio vs the 5-op card codec SNR 50-51 dB, vs the CPU codec 43.9 dB as before; card_codec_s 0.79 -> 0.69 s (fix) |
| (kept off) grouped-MoE bmm configs for M >= 256 only (`QWEN36_GROUPED_BMM_MIN_M=256`): deterministic in 3 runs, voice prefill 1.95 -> 1.68 s, but teacher-forced decode top-1 15/24 (off: 18/24), logits 0.9971 (0.9973) | - | - | - | - |
| 15 codec windows at 3 fixed sizes 32 / 64 / 96 frames (`DRAMA3_CODEC_WINDOWS`, default "32,64,96") with L1_SMALL 65536 (`DRAMA3_L1_SMALL`, default; was the GDN minimum 24576) (`gate22`, `gate22b`) | 23.4 / 10.65 | 1.17-1.22 s | 1.21 (first) / 0.85; close as a later line 0.92 | codes identical to gate21; audio vs gate21 SNR 54-57 dB; card_codec_s 0.69 -> 0.43-0.47 s (fix) |
| 16 one-token GDN output gate without the +-1e4 clips (they never bind at T = 1), silu fused into the multiply as an input activation (`QWEN36_GDN_DECODE_NOCLIP`, default 1 in `run_gen.sh --fast`) (`gate23`) | 23.2 / 10.65 | 1.18 s | 1.20 / 0.84 | golden bit-identical to lever 13 (decode 23.7 -> 23.0 ms/step); frames and audio bit-identical to gate22 |
| 17 tower and backbone share the position / cos / sin buffers (`DRAMA3_SHARED_POS`, default 1): the tower steps at the position the backbone decode just staged, 3 uploads fewer per frame (`gate24`) | 23.1 / 10.4 | 1.18 s | 1.20 / 0.84 | frames and audio bit-identical to gate23 |
| (kept off) attention output gate with the sigmoid as the multiply's input activation (`QWEN36_ATTN_GATE_FUSED=1`) | - | - | - | bit-identical, no measurable gain |
| 18 grouped-MoE gate_up on an 8x8 core grid from M = 256 (`QWEN36_GROUPED_GU_GRID_MIN_M`, default 256 in `run_gen.sh --fast`): ttnn then picks a program that is bit-identical to the default at M = 640 and 1.8x faster (16.8 -> 9.2 ms; `tests/sweep_grouped_grid.py`) (`gate26`) | 23.1 / 10.4 | 1.18 s (first line 3.13 -> 2.83 s) | 1.16 / 0.83 | golden bit-identical twice (prefill 1.95 -> 1.71 s); voice prefill 1.96 -> 1.70 s; frames and audio bit-identical to gate24 |
| (kept off) grouped-MoE matmuls on the 11x10 grid below M = 256 (`QWEN36_GROUPED_SMALL_GRID=1`): gate_up 7.6 -> 1.4 ms at M = 32, deterministic in 3 runs, but suffix cand-logit PCC vs the whole prompt 0.9994 / 0.9987 (default 0.9998 / 0.9998), top-10 overlap 9 on fix | - | - | - | - |
| 19 lean GDN step: q's l2 norm and the K^-0.5 scale in one multiply by 1 / K (`QWEN36_GDN_LEAN_QSCALE`, default 1 in `run_gen.sh --fast`) (`gate27`) | 22.95 / 10.5 | 1.2 s (first 2.88) | 1.17 / 0.84 | golden: decode 23.0 -> 22.9 ms/step, final norm 0.9929 (from 0.9926), logits 0.9974; Whisper OK, frames 138 / 266 |
| (not applied) bit-identical core_grid choices for the 128-row prefill linears (`tests/tune_prefill_linear.py`): 3.6 ms per suffix prefill in total | - | - | - | - |

**Final render (all levers through 19, defaults): `/opt/ep0/day4b/out_v2`, overall RTF 0.853, lines 0.82-0.96, TTFA
1.19-1.23 s after the first line (2.86 s first), backbone 23.0 + tower 10.5 ms/frame.** Earlier renders kept:
`out_v2_first` (evening, RTF 1.71), `out_v2_cp1` (L7, 1.011), `out_v2_cp2` (L12, 0.943), `out_v2_cp3` (L14, 0.89).

Defaults now (all switches in one place): `run_gen.sh --fast` sets `QWEN36_MOE_DECODE_INDEXED=1
QWEN36_ROUTER_TOPK_FIRST=1 QWEN36_GDN_LEAN_DECODE=1 QWEN36_GDN_DECODE_NOCLIP=1 QWEN36_GROUPED_GU_GRID_MIN_M=256
QWEN36_GDN_LEAN_QSCALE=1` (+ the Day 4 fidelity / K-block env); drama3 code defaults on: `DRAMA3_PREFIX_CACHE`,
`DRAMA3_TOWER_DEVICE_SAMPLING`, `DRAMA3_GDN_SPLIT_CONV`, `DRAMA3_GDN_CONV_PARTIALS`, `DRAMA3_SHARDED_DECODE_NORM`,
`DRAMA3_DECODE_LINEAR`, `DRAMA3_ROPE_MATMUL`, `DRAMA3_HOST_OVERLAP`, `DRAMA3_SHARED_POS`, `DRAMA3_CARD_CODEC`,
`DRAMA3_CODEC_SNAKE_FUSED`, `DRAMA3_CODEC_WINDOWS=32,64,96`, `DRAMA3_L1_SMALL=65536`, codec fp32 activations with
bf16-rounded conv inputs. Off (opt-in, see the rows above): `QWEN36_GROUPED_BMM_CONFIG`, `QWEN36_GROUPED_SMALL_GRID`,
`QWEN36_MOE_DECODE_PRESCALE`, `QWEN36_ATTN_GATE_FUSED`, `DRAMA3_CODEC_TRACE`, tower `--tower-mlp bfp8`.
Every qwen36 / experimental change is opt-in by env and single-device only (TP paths untouched).

Multi-seed quality check of the final config (`tests/gate_seeds.py`, seeds 1-5, close + fix, ref_v2): Whisper mean match
close 0.938 / fix 1.000, identical to the Day 4 math (`seeds_base`); frames close 131-142 (Day 4 math 136-146), fix
264-277 (266-281); no clipping (`/opt/ep0/day4b/seeds_final`).

Checkpoint render after lever 12 (`/opt/ep0/day4b/out_v2_cp2`): **overall RTF 0.943** (97.2 s audio in 91.7 s),
lines 0.89-1.07 (the first line 1.07 with the voice prefill; second_opinion's 1.013 included a 1.9 s stall writing
the wav, since then `compute_s` stops before the file write and `save_s` is reported separately), TTFA 1.26-1.38 s
after the first line (3.18 s first). Chip 52-63 W, board 175-207 W.

Checkpoint render after lever 7 (`/opt/ep0/day4b/out_v2_cp1`, 8 lines): overall RTF 1.011 (first line 1.20 incl. the
voice prefill, the others 0.965-1.074), TTFA 1.29-1.37 s after the first line.

Card codec notes: conv1d (k=7, causal via symmetric padding (k-1)d + first L rows) works for all stage shapes
with BLOCK_SHARDED for >= 384 channels, HEIGHT_SHARDED below; k=1 convs as ttnn.linear (conv1d's L1 plan fails
at L=24576 x 384). Transposed convs = k=2 conv1d to s*Cout channels + row-major reshape. **Precision pitfall**:
fp32 activations fed straight into conv1d are truncated to bf16 inside (each k=7 conv shrank the signal by
0.31 %, -0.24 dB at the output, SNR 31 dB); `ttnn.typecast` to bf16 first rounds to nearest (SNR 44 dB).
`tests/check_codec_ops.py` checks every op on host inputs (gain + SNR), `tests/check_codec_device.py` the chain.


For an agent that knows nothing about this work. Server: the test machine (root). Work tree: `/opt/ep0/tt-metal`
(branch `ep0/drama3`, local commits only, never push). Model: `/opt/llm/hf/drama-3` (mounted read-only as
`/models/drama-3` in the container). Background notes: `drama3/NOTES.md` (Days 1-4), `drama3/MAPPING.md` (weights).
Project rules: the repository's `CLAUDE.md` (long jobs in the background with
logs and your own polling, disk precautions, no services, no pushes).

### Day 4b: how the prefix cache works

`FastSession.split_point` finds the rows shared by the template (prompts of two one-letter lines + the real
lines: 677 rows for ref_v2) and cuts back to a multiple of 128 (P = 640) such that row P is shared too (it labels
the prefix's last tower row). 128 = the GDN scan chunk (so chunk boundaries match a whole-prompt prefill) and
the flexible chunked SDPA's chunk. `set_voice` prefills rows[:P] (backbone + tower, start 0), then snapshots the
30 GDN rec/conv states into buffers allocated during warm-up (before any trace capture). Per line
`_prefill_line` restores the GDN state by `ttnn.copy` and prefills rows[P:] at `start=P` (bucket 128): RoPE from
position P, paged K/V into blocks P/64.., `chunk_start_idx_tensor` = P. The prefix K/V (backbone and tower) is
never copied: nothing writes blocks < P/64 after `set_voice`. Warm-up compiles the prefix bucket (640) and the
suffix bucket (128); after the captures it drops the voice so the first line pays (and reports) the voice
prefill. `DRAMA3_PREFIX_CACHE=0` = the Day 4 whole-prompt path.

### Day 4b: real time, where the limit is (measured, `tests/profile_frame.py`)

**Overnight update: the estimate below was too pessimistic.** Real time came without custom kernels: the indexed
top-k MoE decode (-8 ms), tuned one-token matmul program configs (-6 ms), sharded one-token norms, split /
partial-sum GDN conv, lean GDN step, router / RoPE / gate op trims, device-side codebook sampling, host overlap,
and the codec decoder on the card via ttnn.conv1d (73 ms per 96-frame window). The text below is the evening
analysis, kept for the record (it was right that the CPU codec alone could not keep up).

Real time = 21.53 frames/s = **46.4 ms per frame** end to end. Device time per frame, traces replayed back to back:
backbone decode trace **44.0 ms**, fused tower trace **13.1 ms** (10.9 ms of layers + 2.2 ms device sampling)
= **57.1 ms of card time per frame before any host work**. Measured per frame in the pipeline: 45.4 + 17.1 =
62.5 ms (host: main-token sampling 0.56 ms, 5 embedding uploads, staging, readbacks, ~5 ms total).
Backbone pieces (`tests/profile_decode.py`, traced): GDN layer 1.155 ms (mixer 0.563 + MoE 0.588), attention
layer 0.982 ms (mixer 0.388); MoE routed 0.407 ms of which the two sparse matmuls dominate; 18 MB of expert
weights per layer would take ~0.04 ms at the card's DRAM bandwidth, i.e. the decode is op/dispatch bound, not
bandwidth bound. The codec on the host CPU costs 51 ms per generated frame inside Fish's stream windows (it
re-decodes 32 frames of overlap per 64-frame window, 1.5x work) with the generator running; uncontended eager
50 ms, compiled ~31 ms per window frame. Codec decode = 6.76 GFLOP per frame (`tests/codec_flops.py`: decoder
6.43 + quantizer 0.33; 54 M + 260 M parameters). The final window is a 2.5-3.6 s tail per line.
So: even with zero host time and an infinitely fast codec, the card alone is at RTF >= 1.23; with Fish's CPU
codec the CPU alone would be at ~1.1. RTF <= 1 needs BOTH the backbone decode at <= ~30 ms device (custom fused
GDN-decode and MoE-decode kernels: MoE 23.5 ms -> ~10, GDN mixer 16.9 ms -> ~5; estimate, days of kernel work)
AND the codec off the CPU (card codec: ~10 GFLOP per new frame incl. overlap, estimate 2-5 ms card time per frame
op-bound, ~0.6 GB of BF16 weights in the free 6.3 GB, removes the 2.5-3.6 s tail and the CPU contention;
estimate 2-4 days to port the conv / transposed-conv / Snake / windowed-transformer stack to ttnn).

## 1. Current state (Day 4; Day 4b changes in section 0)

Fish Audio Drama 3 text-to-speech runs end to end with a voice clone:

| stage | where | precision |
|---|---|---|
| backbone (Qwen3.6-35B-A3B layout, 40 layers, 256 experts top-8) | p150, paged KV, one masked prefill, traced decode | weights from the tensor cache: experts gate/up BFP4, down BFP8, projections BFP8; prefill HiFi2; decode LoFi (expert decode matmuls fp32-accumulating) |
| codebook tower (13 layers) | p150, 9 traced segments per frame | BF16 weights, HiFi4 + fp32 acc |
| sampling (Fish defaults: T 1.0, top-p 0.9, RAS, codebook top-k 30) | host | fp32 |
| codec (Fish modded DAC, 44.1 kHz) | host CPU, separate process, torch.compile'd, Fish's stream windows | fp32 |

Card DRAM: 27.72 / 34.18 GB (backbone 26.05 incl. 4,097-row candidate LM head, tower 1.68). Power while speaking:
chip 49-62 W, board 176-190 W (idle 19 / 81 W).

Fastest config = `run_gen.sh --fast`, which sets:
`QWEN36_MOE_MATH_FIDELITY=HiFi2 QWEN36_MLP_MATH_FIDELITY=HiFi2 QWEN36_ATTN_MATH_FIDELITY=HiFi2` (prefill),
`QWEN36_DECODE_MATH_FIDELITY=LoFi QWEN36_MOE_DECODE_BLOCK_GATE_UP=32 QWEN36_MOE_DECODE_BLOCK_DOWN=8` (decode),
`TT_METAL_CACHE=/opt/ep0/jit_cache`, PYTHONPATH + `/opt/ep0/pyext-codec`. Defaults inside the code:
`DRAMA3_GROUPED_MOE=1` (grouped MoE prefill), `DRAMA3_TOWER_TRACE=1`, tower `--tower-mlp bf16 --tower-fidelity HiFi4`,
`--codec-threads 6`, seed 0, max 600 frames.

Measured (ms/frame = card backbone + tower; `close`/`fix` with ref_long, seed 0; outputs in /opt/ep0/day4/gateN):

| run | change | backbone | tower | prefill | RTF close / fix | TTFA |
|---|---|---|---|---|---|---|
| Day 3 | host tower, eager concat-KV backbone | 186 | 270 | 10 s + 5 s tail + 4 s tower | 13.7 / 12.5 | 17-20 s |
| gate1 | paged + traced backbone, whole-prompt prefill, card tower eager | 65 | 41 | 10.7 s | 5.2 / 4.2 | 11.3 s |
| gate2 | streaming codec (thread) | 66 | 57-61 | 10.7 s | 5.0 / 4.0 | 12.4 s |
| gate3 | codec in its own process | 65 | 50-52 | 10.7 s | 4.8 / 3.7 | 12.2 s |
| gate4 | grouped MoE prefill | 65 | 52-55 | 2.0 s | 3.4 / 2.96 | 3.4 s |
| gate5 | traced tower | 65.5 | 19.5 | 2.0 s | 2.8 / 2.24 | 3.2 s |
| gate6 | MoE decode K blocking | 45.5 | 19.5-20.3 | 2.0 s | 2.3 / 1.89 | 3.0 s |
| final 8 lines ref_long (`/opt/ep0/day4/out`) | | 45.6 | 20.5 | 2.0 s | overall 1.88 (1.79-2.33) | 2.98-3.02 s |
| final 8 lines ref_v2 (`/opt/ep0/day4/out_v2`) | | 45.6 | 20.5 | 2.05 s | overall 1.91 (1.79-2.51) | 3.05-3.08 s (close 3.8 s, see pitfall 8) |

gate7 = the fixed-length K/V fill (commit 702a67c3): close/fix frames bit-identical to day4/out.

## 2. Exact commands (on the server)

Everything runs in the TT image through `/opt/ep0/run.sh "<cmd>"` (podman, `/opt/ep0` mounted with `:z`,
model at `/models/drama-3`, cwd `/opt/ep0/tt-metal`). D=`/opt/ep0/tt-metal/models/demos/blackhole/drama3`.

- Gate (2 lines) + check:
  `nohup $D/run_gen.sh /opt/ep0/day4/gateN.log --fast --lines close,fix --out /opt/ep0/day4/gateN &`
  then `/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate /opt/ep0/day4/gateN"`
  (Whisper transcript, frame count vs Day 3 ±10 %, identical-frame prefix; writes gate.json).
  Takes ~4-5 min (codec compile 40-95 s, warm-up + captures ~25 s, then ~20 s per line).
- Full render (ref_long and ref_v2, power, SHA256SUMS): `nohup $D/render_day4.sh > /opt/ep0/day4/render.log &`
  (edit the output dirs first if you don't want to overwrite day4/out*). Then
  `/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_day4 <dir> [<dir>]"`.
- Voice golden check of the paged backbone (prefill per layer + 24 teacher frames, traced vs eager):
  `/opt/ep0/run.sh "TT_METAL_CACHE=/opt/ep0/jit_cache QWEN36_MOE_MATH_FIDELITY=HiFi2 QWEN36_MLP_MATH_FIDELITY=HiFi2 QWEN36_ATTN_MATH_FIDELITY=HiFi2 QWEN36_DECODE_MATH_FIDELITY=LoFi QWEN36_MOE_DECODE_BLOCK_GATE_UP=32 QWEN36_MOE_DECODE_BLOCK_DOWN=8 python -m models.demos.blackhole.drama3.tests.check_paged_backbone --out /opt/ep0/day4/x.json"`
  (golden: `/opt/ep0/day3/golden_voice`, rows `/opt/ep0/day3/close.rows_teacher24.pt`).
- Tower check vs host FP32 tower: same env, `python -m models.demos.blackhole.drama3.tests.check_tower_device [--mlp bf16|bfp8] [--fidelity HiFi4]`.
- Profiles: `tests/profile_decode.py` (per-piece traced decode times), `tests/profile_moe_decode.py`,
  `tests/profile_prefill.py`, `tests/profile_moe_prefill.py`, `tests/bench_codec.py`.
- Smoke test of the card: `/opt/ep0/run.sh "python /opt/ep0/smoke.py"` (expect SMOKE_OK, PCC 0.9999).
- **Day 4b** (defaults now: prefix cache on, device sampling on): gate with ref_v2 =
  `nohup $D/run_gen.sh /opt/ep0/day4b/gateN.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day4b/gateN &`,
  then the gate module with `GATE_BASE=/opt/ep0/day4/out_v2` (Day 4 ref_v2 baseline). Several seeds in one
  process: `--seeds 1,2,3` (files tagged `_s<seed>`). Switches via `EXTRA_ENV="..."` before run_gen.sh:
  `DRAMA3_PREFIX_CACHE=0`, `DRAMA3_TOWER_DEVICE_SAMPLING=0`, `DRAMA3_SAMPLER_CHECK=1` (host re-draw of every code,
  counts disagreements in timing.json `sampler_mismatch`), `QWEN36_GROUPED_BMM_CONFIG=1` (broken, see pitfalls).
  Full render: `nohup $D/render_day4b.sh [/opt/ep0/day4b/out_v2] > /opt/ep0/day4b/render.log &`, then summarize_day4.
  `$D/run_check.sh <log> <module> <args>` (copy also at /opt/ep0/day4b/) runs any drama3 module with the --fast env
  (`tests.check_prefix_cache`, `tests.profile_suffix`, `tests.profile_frame`, `tests.check_device_sampler`,
  `tests.check_bmm_config`, `tests.check_paged_backbone`, `tests.profile_decode`, `tests.profile_moe_decode`).
- Day 1/2 references: `reference/cpu_layer_ref.py` (FP32 CPU, layer by layer; `--rows` for frame-row prompts,
  `--quant qwen36` for the device weight formats), `tt/device_check.py --chain`.

Outputs: `<out>/<id>.wav`, `<id>.frames.pt` ([N, 11] = main token + 10 codes), `timing.json`, `gen.log`.
timing.json (scorer format: `{id: {compute_s, ttfa_s, power_w, ...}}`) is written by `host/generate_fast.py` per line:
compute_s = line start .. audio complete (incl. codec tail), ttfa_s = line start .. first streamed codec window
decoded, rtf = compute_s / audio_s, plus the breakdown (prefill_s, backbone_ms_per_frame, tower_ms_per_frame,
codec_decode_s, codec_tail_s, ...). power_w / board_power_w are merged afterwards by `power_log.py merge` from a
`tt-smi -s` sampler (2 s) between start_epoch and end_epoch. The scorer itself (WER + speaker similarity) runs on the
workstation side (`fish-drama3-p150/eval/score.py` in this repo).

## 3. Code map

drama3/ (all new):
- `tt/weight_mapping.py` per-layer checkpoint loader (FP8 dequant, qwen36 single-device remap; `experts=False` on a warm cache)
- `make_shim_ckpt.py` qwen3_5_moe config view at `/opt/ep0/drama3_shim` (transformers doesn't know fish_omni)
- `convert_batches.sh` the tensor cache conversion (Day 2; cache `/opt/ep0/cache/drama3/P150/tensor_cache_bfp8`, 700 files, 25 GB, sums in `/opt/ep0/day2/cache.sha256`)
- `tt/device_check.py` Day 2 teacher-forced/chain checks + `build_layer`, `dram_used_gb`
- `tt/backbone.py` Day 3 eager wrapper (concat KV) — superseded
- `tt/backbone_paged.py` **current backbone**: paged KV, external GDN state, masked-bucket prefill, decode trace, grouped-MoE warm-up
- `tt/tower_device.py` **current tower** on the card (eager `step`, traced `capture`/`step_traced`, `prefill`)
- `host/prompt.py` reference-bracket-v1 rows + FrameEmbedder (mixed embedding)
- `host/tower.py` host FP32 tower + Fish's sampler (still the source of weights, tables and sampling functions)
- `host/codec.py` Fish's DAC loader (shims); `host/stream_codec.py` compiled codec + stream windows (thread and process)
- `host/generate.py` Day 3 pipeline; `host/generate_fast.py` **current pipeline**; `host/transcribe.py` Whisper checks
- `run_gen.sh` (`--fast`), `render_day4.sh`, `power_log.py`, `run_chain.sh`
- `reference/cpu_layer_ref.py`, `tests/*` (checks, gates, profiles, summaries)

Day 4b: `tt/backbone_paged.py` prefill(start=), snapshot_gdn(into=) / restore_gdn; `tt/tower_device.py`
prefill(start=, last=), prepare_fused / capture_fused / step_fused (one tower trace per frame);
`tt/device_sampler.py` (Fish's codebook sampler in ttnn: fp32 topk 32, full-softmax normaliser, cumsum, masks,
argmax(probs / noise) via max + one-hot); `host/generate_fast.py` split_point / set_voice / _prefill_line, `--seeds`,
`--codec-tail-threads`; `host/stream_codec.py` tail_threads; `render_day4b.sh`; `tests/check_prefix_cache.py`,
`profile_suffix.py`, `profile_grouped_small.py`, `sweep_grouped_mm.py`, `check_bmm_config.py`,
`check_device_sampler*.py`, `debug_fused.py`, `profile_frame.py`, `codec_flops.py`.

qwen36 patches (all opt-in; single-device path only; TP untouched):
- `tt/moe/moe.py` single-device rank handling (router/experts 4D, shared expert 3D) — required on one card
- `tt/moe/operations.py`, `prefill.py`, `decode.py`, `mlp.py`, `gdn/gated_deltanet.py`, `attention/gated_attention.py`:
  `QWEN36_{MOE,MLP,ATTN}_MATH_FIDELITY` (prefill), `QWEN36_DECODE_MATH_FIDELITY`, `QWEN36_MOE_DECODE_BLOCK_GATE_UP/_DOWN`
- `tt/moe/prefill_grouped.py` + `experts.py` hooks (`grouped_prefill`, `prefill_valid_len`)

Commits in order: 6e3fe980, d77f6290 (Day 1) · fb97e210, fa1a4ccb (Day 2) · 65557069, 896c9317 (Day 3) ·
ab61a606, 67ad64c3, 64d8ea6d, 702a67c3 (Day 4) · this handoff.

## 4. Pitfalls hit

1. **Never compile while a trace is parked.** Every program shape must run once before `begin_trace_capture`
   (generate_fast `warmup`: grouped-MoE M buckets, the prefill bucket, tower prefill + step, then captures).
   Anything shape-dependent at request time is a bug: per-length K/V block counts (fixed in 702a67c3), free MoE M
   (bucketed 128..768), new prefill buckets (`run` asserts all lines share the warm-up bucket, 768 for 704-734 rows).
2. **Concat KV cache** (Day 3) recompiles per length and the first line is ~3.5x slower; the paged cache fixes it.
3. **No padding rows in GDN state**: a zero row is not neutral (beta = sigmoid(0) = 0.5); use valid_len (masked prefill).
4. **Single-device MoE shapes**: the layer stack carries [1,S,H]; experts need [1,1,S,H]; shared expert reads T from dim 1.
5. **ttnn host conversion grabs the card**: `ttnn.from_torch(..., bfloat4_b)` without a device still opens it and holds the KMD lock; a device job started meanwhile waits.
6. **Codec loading**: `dac/__init__` drags in argbind/encodec/audiotools; `host/codec.py` loads the needed submodules behind namespace stubs. Packages are `--no-deps` in `/opt/ep0/pyext-codec`; faster-whisper lives in `/opt/ep0/pyext-whisper` and must only be on the path of the Whisper process (it shadows transformers' tokenizers/hub).
7. **Codes >= 1024**: residual heads have 4096 outputs, the codec's residual codebooks 1024; masked (as Fish's reference sampler). The device heads keep 1024 rows and pad back to 4096 (-inf) before sampling so the RNG stream matches the host sampler.
8. **Fish's sampler is rank-noise based**: tiny logit changes reorder ranks and change the draw. Codes can only be identical where the math is identical; compare towers with forced codes.
9. **Special-token rows** (`<|im_start|>`, `\n` before `assistant`) sit at PCC ~0.97-0.985 in this voice context (prefill and decode alike); frame rows are clean.
10. **Residual PCC is flattered** by one massive channel (1269, ~2,800; final-norm gain 0.027): judge in the final-norm space (`tests/analyze_chain.py`) and on candidate logits/top-k.
11. **CPU contention**: the codec as a thread costs the tower step ~20 ms (GIL + cores); as a process ~10 ms. The host has 8C/16T.
12. **Codec startup**: torch.compile of the decoder 40-95 s per process (inductor cache `/opt/ep0/inductor_cache`).
13. **SELinux mount**: run.sh now uses `:z` (shared); with `:Z` a second container relabels `/opt/ep0` and breaks the first.
14. **Disk**: the SATA SSD has dropped writes under load. Before/after big writes: `dmesg --level=err | grep -c "I/O error"` (0 so far), `sync; echo 3 > /proc/sys/vm/drop_caches`, then sha256. Keep it under 85 % (now 68 %). JIT cache `/opt/ep0/jit_cache` is 2.3 GB.
15. **Shell quoting via ssh**: apostrophes / f-strings in inline heredocs broke several commands; write scripts to files and scp them.
16. The worktree's gitdir (`/opt/tenstorrent/tt-metal/.git`) isn't mounted in the container: commit on the host.
17. **(Day 4b) Device buffers allocated after a trace capture can overlap that trace's intermediates** and are
    overwritten by its replays (tt-metal warns once: "Allocating device buffers is potentially unsafe ...").
    Everything that must survive a replay (GDN prefix snapshot, the fused tower's embedding tables / noise /
    sampler constants) is allocated in warm-up BEFORE `bb.capture`. The first device-sampling attempt allocated
    them after the backbone capture: every code was wrong and no line ever stopped.
18. **(Day 4b) The grouped-MoE bmm program configs** (`MatmulMultiCoreReuseProgramConfig`, one expert per core,
    full N) are 2-7x faster and match the default bit-close in isolation for every M bucket (PCC 0.999997), but in
    the full backbone prefill the golden check falls to final-norm PCC 0.84 with a NaN layer mean. Cause unknown
    (suspects: L1 placement next to other buffers, the swiglu/embedding inputs' memory configs). Kept behind
    `QWEN36_GROUPED_BMM_CONFIG=1`, off. Always run `tests.check_paged_backbone` (golden) after a prefill change; a
    two-line gate alone did not catch it.
19. **(Day 4b) One seed is not a length gate.** Over seeds 1-5 the Day 4 math gives close 136-146 frames and fix
    266-281 (`/opt/ep0/day4b/seeds_base`); the ±10 % rule against one baseline sample can fail on a math-identical
    change. The prefix cache's suffix split changes the rounding (bf16 GDN state at row 640, chunked SDPA), so the
    main-token draw changes after a few frames: fix keeps 4 identical frames, close 0.

## 5. Open levers, ranked by expected gain (estimates, not measured)

**Day 4b status:** 1 (prefix cache) done; 2 (tower round trips) done as device sampling (20.3 -> 17.1 ms; the
tower trace itself is 13.1 ms device). New ranking for whoever continues:
a. Backbone decode kernels (44.0 ms device): MoE decode sparse matmuls (gate_up 32 cores at 1 N-tile each) and the
   256-expert glue, the GDN decode mixer (0.56 ms x 30). Only a fused-kernel effort moves this by >10 ms.
b. Card codec (removes the 2.5-3.6 s tail per line and the CPU codec ceiling of ~47-51 ms per frame; see section 0).
c. The bmm program configs (pitfall 18), once understood: voice prefill 1.96 -> 1.59 s, suffix 0.75 -> 0.60 s.
d. TTFA below 1 s: the suffix GDN chunk prefill (12 ms per layer x 30 = 0.38 s at 128 rows, ~40 glue ops per
   layer) and Fish's first stream window of 10 frames (~0.6 s at 62 ms/frame); a smaller first window deviates
   from Fish's stream defaults.
e. Host overlap: pre-draw the next frame's noise while the backbone runs (RNG order is value-independent),
   embed the backbone input on the card; ~1-3 ms per frame.

Day 4 list (kept for reference):

1. **Voice-reference prefix cache** (TTFA; requested by Sebastian): ~690 of ~720 prompt rows are identical for every
   line of one voice. Prefill the prefix once, snapshot GDN rec/conv (30 layers, `ttnn.clone`), keep the paged KV
   blocks of the prefix (10 backbone + 13 tower layers), restore per line by on-device copy and prefill only the suffix
   with `chunk_start` = prefix length (block-aligned to 64; qwen36's masked prefill supports chunk_start /
   `chunk_start_idx_tensor` and carried GDN state). Check where the line text begins: the prefix ends after
   `<|im_end|>\n<|im_start|>user\n` (prompt.py builds segments separately, so the ids are identical); move the split
   back to a 64-row boundary. Expected per-line TTFA ~0.8-1.2 s (suffix bucket 128 + 10 frames + first codec window),
   first line unchanged (~3 s); per-line RTF -0.15. Gate: same seed with/without cache -> logits PCC at the first
   frame and codes (identical only if the suffix prefill's math matches; otherwise explain).
2. **Fewer host round trips in the tower** (20 ms -> ~8-10 ms): upload code indices and look the embeddings up on the
   card; sample on device (top-k 30 + uploaded noise would keep Fish's semantics); or merge segments.
3. **Backbone op overhead** (45.6 ms; profile_decode: GDN mixer 0.56 ms x 30 = 17 ms, MoE ~0.6 ms x 40 incl. router
   0.12 ms, attention 0.39 ms x 10): DRAM-sharded decode matmul configs for the GDN/attention projections, a cheaper
   router (topk/scatter), trimming the MoE glue ops (permute/reshape on 256-expert tensors). Maybe -10 to -15 ms.
4. **Codec** (~30 ms CPU per frame + 2-3 s tail per line): it is the floor for RTF ~1 on this CPU; a card codec would
   remove it (large port). Cheaper: adaptive windows (decode whenever idle) to cut the tail ~1 s (deviates from Fish's
   fixed windows).
5. **Overlap tower and backbone**: the next frame's backbone input needs only main + code1, so its decode could start
   after head 0 while the host samples codes 2-9 (host/device overlap, a few ms).
6. Tower BFP8: tested, no gain (round-trip bound), slightly worse accuracy — skip.
7. Grouped MoE prefill: layers with one hot expert (~580 rows) force M = 640 for all 256 experts; treating hot experts
   separately could halve the remaining ~1.4 s.

## 6. Quality gates and thresholds

- Per speed change: re-render `close` + `fix` (ref_long, seed 0): Whisper transcript intelligible/unchanged
  (close: "moat" is always heard as "mode", that is fine), frame count within +-10 % of Day 3 (close 128, fix 254),
  codes bit-identical where the math is meant to be identical (trace vs eager, scheduling-only changes).
- Backbone numerics vs the CPU FP32 golden: per-layer PCC >= 0.99 over real rows; same top candidate at the
  first-frame row; candidate-logit PCC ~0.995+; teacher-forced decode logits PCC mean ~0.997.
- Tower vs host FP32 tower (forced codes): per-layer PCC >= 0.999, head logits >= 0.996, head top-1 agreement ~93 %.
- Final: all 8 lines, no clipping, Whisper match vs the written form as Day 3 (misses only on "Tenstorrent",
  "Tensix"), the scorer's WER + speaker similarity on the Windows side.
