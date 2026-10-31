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

