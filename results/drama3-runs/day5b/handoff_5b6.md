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
