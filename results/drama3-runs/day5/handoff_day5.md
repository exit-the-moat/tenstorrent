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
