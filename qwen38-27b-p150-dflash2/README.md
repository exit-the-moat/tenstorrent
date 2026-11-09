# Qwen3.8-27B with DFlash2 speculative decoding on 1 or 2 Tenstorrent p150a

Serve [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the DFlash2 drafter on one or two p150a cards, using
the public `tt-inference-server` image and one switch per setup:

    ./serve.sh --cards 2 --models $MODELS --cache $CACHE    # 2 x p150a, up to 8 requests at once
    ./serve.sh --cards 1 --models $MODELS --cache $CACHE    # 1 x p150a, one request at a time

`--cards N` sets one environment variable, `QWEN38_BEST=2card` or `1card`; all the switches behind it are listed in one
file (`qwen36/ep2_dflash/df_best.py`). The server is OpenAI-compatible on `http://127.0.0.1:8087`. Measured numbers
and their conditions are below; `check/` holds the scripts that made them.

Episode 2 of the [Exit the Moat](https://www.youtube.com/@ExitTheMoat) channel. Not affiliated with or endorsed by
Tenstorrent, Alibaba Cloud (Qwen) or the DFlash authors.

## What you need

- **Cards:** one or two Tenstorrent p150a. Two cards need **two QSFP-DD cables** between them (the 4 direct links of
  the `P150_X2` setup; this package was only gated with both cables). Power each card with its own native 12V-2x6 cable
  from the PSU; a PCIe-to-12V-2x6 adapter made our PSU shut off at power-on.
- **Host:** Linux, one x86-64 CPU (we used 8 cores / 16 threads, 62 GB RAM, PCIe Gen3 x16 + x4). `podman` run as root
  (rootful; the card and hugepage devices are passed through). `tt-smi` on the host (`pip install tt-smi`) for resets.
- **Driver and firmware** (as measured): [tt-kmd](https://github.com/tenstorrent/tt-kmd) **2.11.0**, firmware bundle ([tt-firmware](https://github.com/tenstorrent/tt-firmware)) **19.15.0**. Other versions may work; we did
  not test them.
- **Hugepages:** 1 GB hugepages, 4 per card, mounted at `/dev/hugepages-1G`. Check with
  `cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages`; `grep Huge /proc/meminfo` shows only the 2 MB pool and can read 0 on a correct setup. The
  `tenstorrent-tools` package ([tt-system-tools](https://github.com/tenstorrent/tt-system-tools)) sets this up; if your distribution has no package, use that repository's scripts or set the hugepages up yourself.
- **Disk:** about 54 GB for the model, 3.6 GB for the drafter, and a tensor cache of about 23 GB for the 2-card setup
  plus 23 GB for the 1-card setup (separate folders inside `--cache`, written once). A slow SATA SSD is fine; see
  `--write-bps` below.

## Set up

All commands run from this folder. Pick two directories:

    export MODELS=/path/to/models CACHE=/path/to/cache

The image runs as an unprivileged user: this package's folder and `$MODELS` must be readable by other users. `serve.sh`
makes the cache and work folders writable for that user itself.

1. **Weights.** `pip install -U huggingface_hub`, then:

       hf download Qwen/Qwen3.8-27B --local-dir $MODELS/qwen38-bf16
       hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir $MODELS/qwen38-dflash2

   The folder names matter: `serve.sh` expects exactly these two. The first holds about 54 GB in 18 shards, the second
   one `model.safetensors` of 3.6 GB.

2. **Image** (public; about 15 GB):

       podman pull ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af

3. **Check the cards.** `tt-smi` lists both, then:

       ./serve.sh smoke 0 1      # expect SMOKE_OK for each card

## Start

    ./serve.sh --cards 2 --models $MODELS --cache $CACHE

or `--cards 1`, which uses card 0 unless you add `--device 1`. The script waits until the server answers `/health` with
`ready` (it prints `READY`; the log is `work/stats/server.log`). The first start of each setup builds its tensor cache and
compiles the kernels; later starts skip both. Measured with `--write-bps /dev/sda:80mb` on a SATA SSD: 2 cards 24 min first,
6 min later; 1 card 17 min first, 4 min later (later starts with the weights still in the page cache). With a ready tensor
cache but no compiled kernels (a new `--work` folder) expect about 12 min on 2 cards and 6 min on 1 card. While warming up,
`server.log` may show a `BrokenPipeError` from `/health` when a poll times out; that is harmless. Each start overwrites `server.log`.

Options (`./serve.sh --help`): `--port`, `--name`, `--work` (logs and kernel cache, default `./work`), and for a slow
disk `--write-bps /dev/sda:80mb` (throttles the container's writes; we ran with it on a SATA SSD). The device must be the
physical disk (for example `/dev/sda` or `/dev/nvme0n1`), not the `/dev/mapper/...` name `df` shows on LVM.

**Try it:**

    curl -s 127.0.0.1:8087/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen38",
      "messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":128,
      "temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'

**Chat in a browser** (standard library only, shows first-token time and decode speed separately for every reply):

    python3 chat/chat.py --no-tunnel                  # the model server runs on this machine
    python3 chat/chat.py --ssh-host myserver          # the model server runs on another machine (opens an ssh tunnel)
    # then open http://127.0.0.1:8790
    # --ssh-host needs a working key-based ssh login. Stop it with Ctrl+C, which also closes the tunnel.

Any OpenAI-compatible client works against `127.0.0.1:8087/v1` (model name `qwen38`; greedy decoding only, `temperature`
must be 0). Thinking is on unless you send `chat_template_kwargs: {"enable_thinking": false}`.

## Stop, and what to do when a card wedges

    ./serve.sh stop

prints `CLEAN_STOP` when the server closed the mesh properly. After `CLEAN_STOP` both cards are usable right away.

- If the stop does **not** print `CLEAN_STOP`, or a server crashed: the 2-card setup can leave ethernet cores on the
  cards active, and the next start then fails with `Timed out while waiting for active ethernet core ...`. Reset both
  cards (**both**, never one alone): `tt-smi -r 0 1`, then `./serve.sh smoke 0 1`.
- If a card stays wedged after that, soft-reboot the host. Nothing in this package needs a firmware change.

In the runs below every stop printed `CLEAN_STOP` and both cards passed the smoke test right after, on both setups, with
no reset. Reset both cards together; we have not tested resetting one alone as safe.

## What it does

`QWEN38_BEST` is read once at the top of `df_server.py`. Each switch is set with `setdefault`, so an explicit
`--env KEY=VALUE` to `serve.sh` overrides it (the log prints every override). `python3 qwen36/ep2_dflash/df_best.py --env
2card` prints the whole set.

| Setup | `QWEN38_BEST` | What runs |
|---|---|---|
| 2 cards | `2card` | 2 x p150a, tensor-parallel (TP=2). DFlash2 with the verify pass as a batch; up to 8 streams with the multi-stream schedule (speculate while at most 6 streams are live, then a batched plain lane), prefix cache, three drafter / LM-head levers (`DF_DRAFT_LM_HALF`, `QWEN36_PF_LMHEAD`, `DF_DRAFT_BF4ALL`). |
| 1 card | `1card` | 1 x p150a through the two-card code path on a 1x1 mesh (`QWEN36_TP1=1`), DFlash2, one request at a time, 40,960 tokens of context. The drafter is traced, but not chained on the card into the verify input (`DF_CHAIN=0`); mixer weights prefetched too. |

**Patches over the image.** `serve.sh` mounts `qwen36/` over the image's
`models/demos/blackhole/qwen36` (the model code, Apache-2.0, modified: every addition is behind a default-off switch
and only `QWEN38_BEST` turns them on) and, for the 2-card setup only, `patches/model_config.py` over the image's
`models/tt_transformers/tt/model_config.py` (one change: `supported_batches` gets 64 and 128, used by the multi-stream
drafter). Nothing else in the image changes.

Weights: the model is converted to BF4 for the math and uses BF8 KV pages (`QWEN36_NAIVE_BF4`, `QWEN_SDPA_BF8`) on first start.
The original bf16 checkpoint stays untouched.

## What we measured

Measured on 9 November 2026: tt-kmd 2.11.0, firmware 19.15.0, image `0.20.0-de59f8a-03fa3af`, one host with 8 cores / 16
threads, PCIe Gen3 x16 and x4, the card under test alone (only `serve.sh` running). "decode" is tokens after the first
over the time after the first; TTFT is first streamed token minus send. Both are measured separately on
**the same corpus**, `check/prompts.json` (8 code prompts of about 580 tokens), 256 tokens out, greedy, thinking off,
one request at a time.

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

Run them yourself on a started server (assume `--cards 2` here; for one card use `--workers 1` and `reference/1card-exact.json`):

    python3 check/exact.py capture my2 --url http://127.0.0.1:8087
    python3 check/exact.py compare reference/2card-exact.json runs/my2.json     # expect EXACT 5/5
    python3 check/speed.py run my2 --url http://127.0.0.1:8087
    python3 check/gsm8k.py run --config my2 --workers 8                         # 2 cards: --workers 8; 1 card: --workers 1
    python3 check/niah.py my2                                                   # expect NIAH 6/6
    python3 check/prefill.py my2

`check/exact.py compare` needs the same outputs token for token. We got identical outputs across starts and from an
empty cache; we have not tried a second machine. A different image or weight file can change them. GSM8K here is the first 60 test questions (bundled), with thinking
off; expect 59–60 of 60.

## Limits worth knowing

- One card serves **one request at a time**. Two cards serve up to 8 at once; with more than 6 live requests the server
  falls back to the plain lane without speculation for all of them until they drain.
- Context: 2 cards 65,536 tokens per stream (pool 524,288), 1 card 40,960 (49,152 started in development and stayed
  exact but was not gated; 65,536 ran out of DRAM).
- Speed depends a lot on how predictable the text is. The numbers above are code. Prose is markedly slower (on one card, in
  development with the drafter traced but before the mixer prefetch: about 44 tok/s median versus 80 on code); that gap is DFlash2 having less to accept.
- Greedy decoding only (`temperature: 0`).
- **One-card prefill.** `1card` runs the two-card code path on one card, and its prefill is that path's prefill, in the table
  above. It contains **none** of the separate one-card path's prefill work (the `QWEN36_1C_*` switches: explicit prefill
  matmul configs, the fused delta-rule op, traced masked buckets). That work lives on a different code path, was never
  combined with DFlash2 or gated with it, and is not in this package. For orientation only, on other prompts and not
  gated here: that path reached first tokens of 0.60 s at 1k, 2.0 s at 4k, 3.8 s at 8k and 16.2 s at 32k.
- **One-card decode depends on the corpus.** The 75.7 tok/s above is the mean over `check/prompts.json`. On a second code corpus, not in this package, the same server
  measured 79.6-80.2 tok/s.
- **Concurrency.** The speed tables are one request at a time. Concurrency in this gate means GSM8K with 8 parallel clients on
  two cards (59/60, the same texts as the unfolded start). On 2 cards with the prompts of `check/prompts.json` (256 tokens out, requests started together, ad-hoc script not in `check/`, mean of 3 to 5 rounds): 4 at once 56.2 tok/s per stream (49-62), 225 tok/s total while decoding, 191 including the wait for the first token, first token 0.7 s median and 1.2 s for the last; 8 at once 26.4 tok/s per stream (24-32), 211 total, 185 including the wait, first token 1.3 s median and 2.6 s for the last. Above 6 live streams the server stops speculating and decodes plain. Long, less predictable prompts are slower: with 4k slices of source code, 4 at once gave 30.8 tok/s per stream.

## Layout

| Path | What |
|---|---|
| `serve.sh` | start / stop / smoke; every mount and environment variable is in this one script |
| `qwen36/` | the model code mounted over the image's `models/demos/blackhole/qwen36` |
| `qwen36/ep2_dflash/df_best.py` | the two switches, `QWEN38_BEST=2card` / `1card`, every flag in one place |
| `qwen36/ep2_dflash/df_server.py` | the OpenAI-compatible DFlash2 server |
| `patches/model_config.py` | the image's `tt_transformers/tt/model_config.py` with the one-line change (2 cards only) |
| `chat/chat.py` | a minimal browser chat (standard library only) |
| `check/` | `exact.py`, `speed.py`, `gsm8k.py` + `gsm8k60.jsonl`, `niah.py`, `prefill.py`, `prompts.json`, `smoke.py` |
| `reference/` | `2card-exact.json`, `1card-exact.json`: the outputs of our gated runs |
| `results/gate-summary.md` | the numbers of the gated runs, generated from the run logs |

## Licences

- This package's code: Apache-2.0 (see `LICENSE` at the top of this repository). `qwen36/` and `patches/model_config.py`
  are modified versions of Tenstorrent `tt-metal` files (Apache-2.0).
- Qwen3.8-27B: Apache-2.0 (Alibaba Cloud; see the model's `LICENSE`). DFlash2 drafter: Apache-2.0 (see the model card).
  Weights are not part of this package.
- `check/gsm8k60.jsonl`: the first 60 test questions of [GSM8K](https://github.com/openai/grade-school-math) (MIT).
