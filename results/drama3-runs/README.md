# Drama 3 run files (Episode 0)

Unedited `timing.json` / `sanity.json` / `power.txt` / `golden.json` files, logs and run scripts from the runs behind
the tables in `fish-drama3-p150/README.md` and `fish-drama3-p150/drama3/HANDOFF.md`. Paths inside them
(`/opt/ep0/...`) are the work directory on the test machine. Audio and frame files are not included.

| Folder | Run |
|---|---|
| `scoreboard-2026-10-30.md` | the generated scoreboard rows (day 3 to day 6) with the source file of each row |
| `day3/` | first end-to-end run (RTF 14.743) |
| `day4/`, `day4b/` | traced pipeline; `day4b/out_v2` = the end of "day 2" in the video, RTF 0.853 (`--fast` reference) |
| `day5/`, `day5b/` | BFP8 expert test (rejected), first static batching |
| `day6/` | the first narration on the card (38 lines, RTF 0.961), sequential and batched |
| `day7/` | the retake picks before the freeze |
| `load/` | load times: cold (SATA SSD) and warm (page cache), PCIe bandwidth |
| `v17pw/` | `--interactive` after port6 with tt-smi power sampling |
| `power-v11-phase1/` | the power chart in the video: tt-smi samples while the card spoke 12 lines (`lines.csv` = per-line windows) |
| `port5/final8c` | `--interactive` after port5 (8 lines) |
| `port6/i_qdt` | `--interactive` after port6 (grouped-MoE matmul glue, quantizer on the card) |
| `port6/i_final` | port6 base check (bit-identical to port5) |
| `port6/golden.json` | golden check with the `--interactive` levers |
| `port6/lad3/c<B>` | continuous batching, B slots, 16 callers arriving 1 s apart (`bench/ladder.sh`) |
| `port6/lad3/s<B>` | static batching, B streams (`bench/ladder.sh`) |
| `par/b<B>` | static batching before the batch-aware kernels (2026-10-30 morning) |
| `port7/` | same-precision levers for one voice, the frame-time breakdown |
| `port8/` | device profiles (`prof_*`) and the S8 lever set |
| `port9/` | continuous batching up to B = 8, the throughput baseline |
| `port10/i_fin` (= `k_on_1`, repeat `k_on_2`) | **today's `--interactive`**: RTF 0.381, first audio 0.409-0.472 s |
| `port10/i_finpw` | the same with tt-smi power and DRAM sampling |
| `port10/f_short_<B>`, `f_long_<B>`, `fb_*` | full-queue throughput at B slots (C4 levers / port9 base); `*.wcheck.log` = Whisper check |
| `port10/lim.log` | B = 27 / 32 limit runs (out of DRAM, core grid) |
| `narration-v19/`, `narration-v20/` | the episode narration on the card (timing and Whisper analysis per take, no audio); `narration-v20/used_takes.json` lists the 47 takes in the video |

`../rtx-pro-6000-reference/` holds the original model on a rented RTX PRO 6000 (same 8 lines and voice).
