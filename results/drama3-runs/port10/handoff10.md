## Day 8e / port10 (2026-10-31): batched throughput, B up to 24. Read this first

**Status.** Every lever is opt-in and committed locally on `ep0/drama3` (commit ebfac5d3b and the codec follow-up commit). Nothing was pushed. Defaults are unchanged:
- `--fast` is 8/8 identical to `day4b/out_v2`, and `--interactive` 8/8 to `port6/i_qdt` (`/opt/ep0/port10/r_def_10a`, `i_base_10a`, `defs_10b.log`).
- cbatch B 8 with the base flags: frames 32/32 identical to port9 `q_short_8`.

The results are summarised in `fish-drama3-p150/README.md`.

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
- The attention pre kernel (4 cores per row) limits B to 27 (110 worker cores).

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

