# Gate summary (generated from the run logs, 9 November 2026)

### Decode and first token (`check/speed.py`)

Two rounds over the 8 prompts, 16 requests per run. Each run is a separate server start.

| setup | run | decode tok/s, mean (min-max) | TTFT, median | accepted drafts per step | GSM8K | NIAH |
|---|---|---|---|---|---|---|
| 2 cards | unfolded reference | **140.07** (134.8-148.2) | 0.267 s | 4.957 | 59/60 | 6/6 |
| 2 cards | `--cards 2`, start 1, new empty cache | **140.36** (134.4-148.5) | 0.267 s | 4.957 | 59/60 | 6/6 |
| 2 cards | `--cards 2`, start 2 | **140.21** (134.6-148.4) | 0.266 s | 4.957 | 59/60 | 6/6 |
| 1 card | unfolded reference | **75.74** | 0.388 s | 4.809 | 60/60 | 6/6 |
| 1 card | `--cards 1`, start 1, new empty cache | **75.80** | 0.387 s | 4.809 | 60/60 | 6/6 |
| 1 card | `--cards 1`, start 2 | **75.74** | 0.388 s | 4.809 | 60/60 | 6/6 |

### The folded switches against the unfolded starts

`--cards N` must give the same outputs as the same code started with every flag spelled out.

| run | exact, 5 prompts x 128 tokens | GSM8K texts identical |
|---|---|---|
| `--cards 2` start 1 vs unfolded | 5/5 | 60/60 |
| `--cards 2` start 2 vs unfolded | 5/5 | 60/60 |
| `--cards 2` start 2 vs start 1 | 5/5 | 60/60 |
| `--cards 1` start 1 vs unfolded | 5/5 | 60/60 |
| `--cards 1` start 2 vs unfolded | 5/5 | 60/60 |
| `--cards 1` start 2 vs start 1 | 5/5 | 60/60 |

### Prefill (`check/prefill.py`)

Median of 3 requests after one warm-up, no shared prefix. The tok/s column is prompt tokens / TTFT, so it includes the fixed cost per request.

| prompt tokens | 2 cards TTFT, start 1 / start 2 | 2 cards tok/s | 1 card TTFT, start 1 / start 2 | 1 card tok/s |
|---|---|---|---|---|
| 982 | 0.268 / 0.268 s | 3669 | 0.387 / 0.388 s | 2536 |
| 4054 | 0.894 / 0.891 s | 4537 | 1.509 / 1.498 s | 2686 |
| 8150 | 1.811 / 1.808 s | 4501 | 3.099 / 3.096 s | 2630 |
| 32726 | 8.261 / 8.275 s | 3962 | 14.517 / 14.486 s | 2254 |

1-card decode on a second code corpus (2 rounds per run, medians): ref1 79.7, 79.59 tok/s; pk1a 79.56, 79.71 tok/s; pk1b 80.2, 79.84 tok/s.
