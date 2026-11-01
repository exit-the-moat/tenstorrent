# Episode 1: Qwen3.8-27B on 2 × Tenstorrent p150a (TP=2)

Code and data behind Episode 1 ("Qwen 3.8 27B on 2 Tenstorrent p150: getting started"): serving the model on two
p150a cards with vLLM-TT, the team benchmark, the GSM8K check and the patches that got two cards to start.

**Status (2026-11-01): branch `episode-1`, not merged and not public.** It goes to `main` when the episode is
published. Open before that: the `QWEN36_GDN_UNFUSED_RS` numerics fix (`patches/REVIEW.md`, finding 1) and a re-run
of GSM8K with it. Upstream only as an issue, after review.

## Layout

| Path | What |
|---|---|
| `serve/start-ep1.sh` | starts the stock `vllm-tt-metal` 0.20.0-de59f8a image on 1 or 2 cards (port 8085); `EXTRA_ENV` passes the patch switches |
| `serve/p150_x2_1cable_mesh_graph_descriptor.textproto` | the 2-card mesh with **one** cable (`channels { count: 2 }`); the stock start expects 4 eth links |
| `bench/bench.py` | the team benchmark: real source code as prompts (no shared prefix), 1 / 4 / 8 users at once, greedy, thinking off, ≥ 256 tokens out; TTFT, decode tok/s per user, aggregate, prefill; appends to `results.csv`, raw per-request data in `bench/runs/` |
| `eval/gsm8k.py` | first 60 GSM8K test questions, greedy, thinking off; answers in `eval/runs/` |
| `patches/` | `gdn_tp_v5.py` (mounted over `gdn/tp.py`), `bf4/` (the naive all-BF4 variant), `orig/` (the image's own files), `*.diff`, `REVIEW.md` |
| `scripts/` | the night's run scripts as they ran (`run_night*.sh`, `run_1cable.sh`, `run_bf4.sh`, power pass), weight checks |
| `../results/qwen38-27b/` | `results.csv` (bench), `quality.csv` (GSM8K) |

## Two cards

The stock image fails five times in warm-up on 2 × p150a (the supported setups are 4 or 8 cards). Start 6 runs with
`patches/gdn_tp_v5.py` mounted over `models/demos/blackhole/qwen36/tt/gdn/tp.py` and

    -e QWEN36_GDN_CONV1D=0 -e QWEN36_GDN_PREFILL_DRAM=1 -e QWEN36_GDN_UNFUSED_RS=1

plus host sampling. All three switches are prefill-only and default-off (no behaviour change when unset). After
stopping a 2-card server both cards need a soft reboot (ethernet cores stay stuck).

## Measured (2026-11-01, plain decoding, no speculative decoding)

| | 1 × p150a | 2 × p150a |
|---|---|---|
| 1 developer, 1k prompt | 7.4 tok/s | 14.6 tok/s |
| 8 developers, 4k prompts | – (batch 1 only) | 11.2 tok/s each, 58 total |
| GSM8K (60) | 59 / 60 | 59 / 60 |

One cable runs within 1 % of two. Exact rows: `results/qwen38-27b/results.csv`.
