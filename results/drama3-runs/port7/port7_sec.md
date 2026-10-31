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

