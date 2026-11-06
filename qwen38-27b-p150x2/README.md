# Episode 1: Qwen3.8-27B on 1 and 2 × Tenstorrent p150a (TP=2)

Code and data behind Episode 1 ("Qwen 3.8 27B on 2 Tenstorrent p150: getting started"): serving the model on one or
two p150a cards with vLLM-TT, the team benchmark, the GSM8K check and the patches that got two cards to start.

**Status: branch `episode-1`, not merged yet.** Open: the `QWEN36_GDN_UNFUSED_RS` numerics fix (`patches/REVIEW.md`,
finding 1) and a GSM8K re-run with it. Upstream only as an issue, after review.

## Measured (2026-11-01, plain decoding, no speculative decoding)

| | 1 × p150a | 2 × p150a |
|---|---|---|
| 1 developer, 1k prompt | 7.4 tok/s | 14.6 tok/s |
| 8 developers, 4k prompts | – (batch 1 only) | 11.2 tok/s each, 58 total |
| GSM8K (60) | 59 / 60 | 59 / 60 |

One cable runs within 1 % of two. Exact rows: `../results/qwen38-27b/results.csv`. A clean re-run from this folder
(2026-11-05) got 7.43 (1 card), 14.4 and 11.1 / 58.0 (2 cards) and 59 / 60.

## Run it yourself

**You need:** Linux with tt-kmd and 1 GB hugepages (`/dev/hugepages-1G`), podman, Python 3 on the host (bench and
eval use the standard library only), 1 or 2 p150a (two cards joined by QSFP-DD cables), about 54 GB for the weights
and 23 GB per mesh for the tensor cache (one card and two cards are separate caches: 46 GB for both runs). Measured
with firmware 19.15.0 and tt-kmd 2.11.0. `tt-smi` on the PATH is optional (power columns).

All commands run from this folder (`qwen38-27b-p150x2/`). Pick a work folder and a models folder:

    export WORK=$PWD/work MODELS=/path/to/models

1. **Weights** (54 GB): `hf download Qwen/Qwen3.8-27B --local-dir $MODELS/qwen38-bf16` (`pip install -U
   huggingface_hub` gives the `hf` command). The start script expects that folder name and uses the model's
   `chat_template.jinja`. Optional check against the Hugging Face file list:
   `python3 scripts/verify_hf.py Qwen/Qwen3.8-27B $MODELS/qwen38-bf16`; a `FAILED` that lists only `README.md` or
   `.gitattributes` is harmless (the weight shards and configs are what count).
2. **Image:** `podman pull ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af`
3. **Prompt corpus** for the benchmark (real source code): `git clone https://github.com/tenstorrent/tt-metal $WORK/tt-metal`
   and `git -C $WORK/tt-metal checkout 70222e907` (the commit our runs read; any source code works for speed).
4. **Start, one card** (one card serves one request at a time; the 2-card defaults run out of DRAM):

       MODELS=$MODELS CACHE=$WORK/cache LOGDIR=$WORK MESH=P150 DEVS=0 LEN=16384 SEQS=1 POOL=24576 serve/start-ep1.sh

   **Start, two cards** (both cables plugged in): the stock image fails in warm-up on 2 × p150a (supported are 4 or
   8 cards). It runs with our patch `patches/gdn_tp_v5.py` mounted over the image's `gdn/tp.py`, three switches and
   host sampling (the script's default):

       T=/home/container_app_user/tt-metal/models/demos/blackhole/qwen36/tt
       EXTRA_ENV="-v $PWD/patches/gdn_tp_v5.py:$T/gdn/tp.py:ro -e QWEN36_GDN_CONV1D=0 -e QWEN36_GDN_PREFILL_DRAM=1 -e QWEN36_GDN_UNFUSED_RS=1" \
         MODELS=$MODELS CACHE=$WORK/cache LOGDIR=$WORK serve/start-ep1.sh

   The three switches are prefill-only and default-off (no behaviour change when unset). **One cable instead of two:**
   add `-v $PWD/serve/p150_x2_1cable_mesh_graph_descriptor.textproto:/home/container_app_user/p150_x2_1cable.textproto:ro
   -e TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/p150_x2_1cable.textproto` to `EXTRA_ENV` (the stock mesh
   description expects 4 links; one cable has 2).
5. **Wait** until the log (`$WORK/ep1-vllm-*.log`) shows `Application startup complete`: the first start takes
   14-18 minutes (it builds the tensor cache; we measured 14.5 min on one card, 17.3 on two), later starts are faster.
6. **Test:**

       curl -s localhost:8085/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model":"qwen38","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"chat_template_kwargs":{"enable_thinking":false}}'

7. **Benchmark** (the episode's rows: exactly 256 tokens out per request, the default):

   Two cards:

       python3 bench/bench.py --corpus $WORK/tt-metal --config my-2card --system "2x p150a" --cards 2 --cables 2 \
         --precision default-bfp8-bfp4mlp --users 1,8 --inputs 1024,4096

   One card (one user only; `EP1_POWER_CARDS=0` sums the power of card 0 only, otherwise tt-smi adds the idle card):

       EP1_POWER_CARDS=0 python3 bench/bench.py --corpus $WORK/tt-metal --config my-1card --system "1x p150a" \
         --cards 1 --cables 0 --precision default-bfp8-bfp4mlp --users 1 --inputs 1024,4096

   Results go to `$WORK/results.csv` and `$WORK/runs/` (`--results` and `--run-dir` override); `--config` and
   `--system` are just labels in the rows. `--open` lets answers run to their end instead (3-5k tokens, not
   comparable).
8. **GSM8K** (first 60 test questions, greedy, thinking off):
   `python3 eval/gsm8k.py run --config my-2card --precision default-bfp8-bfp4mlp --workers 8` (about 4 minutes on
   two cards; on one card use `--config my-1card`: it answers one question at a time, about 45 minutes). Answers and
   `quality.csv` go to `$WORK/eval` (or `EP1_EVAL_DIR`).
9. **Stop:** `podman stop ep1-vllm`. After a one-card run no reboot is needed (we started two cards right after).
   **After a 2-card run, reboot the host before the cards are used again.** Otherwise the next job fails with
   `Timed out while waiting for active ethernet core 27-25 to become active again. Try resetting the board.` (the
   cards' ethernet cores stay active after the server stops; a soft reboot clears it). Check after the reboot:
   `tt-smi` lists both cards, and the next server start comes up.

Other start-script settings (container name, port, a write throttle for slow disks) are in the header of
`serve/start-ep1.sh`.

## Layout

| Path | What |
|---|---|
| `serve/start-ep1.sh` | starts the stock `vllm-tt-metal` 0.20.0-de59f8a image on 1 or 2 cards (port 8085); settings in its header |
| `serve/p150_x2_1cable_mesh_graph_descriptor.textproto` | the 2-card mesh with **one** cable (`channels { count: 2 }`) |
| `bench/bench.py` | the team benchmark: real source code as prompts (no shared prefix), 1 / 4 / 8 users at once, greedy, thinking off, exactly 256 tokens out; TTFT, decode tok/s per user, aggregate, prefill; appends to `results.csv`, raw per-request data in `runs/` |
| `bench/runs/` | the raw per-request records of our 2026-11-01 runs |
| `eval/gsm8k.py`, `eval/gsm8k60.jsonl` | the GSM8K check and its fixed 60-question subset; our answers in `eval/runs/` |
| `patches/` | `gdn_tp_v5.py` (mounted over `gdn/tp.py`), `bf4/` (the naive all-BF4 variant), `orig/` (the image's own files), `*.diff`, `REVIEW.md` (review findings, open items) |
| `scripts/` | **history:** the night's run scripts as they ran on our server (`/opt/ep1` paths, card locks; not entry points), plus the weight checks `verify_hf.py` / `verify_crc.py` |
| `bench/bench.py.exact-2026-11-01` | history: the bench version the night's scripts called (it pinned the output to 256 tokens; now the default of `bench.py`) |
| `../results/qwen38-27b/` | `results.csv` (bench), `quality.csv` (GSM8K) |
