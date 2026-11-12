# Qwen-Image-2512 on 2 Tenstorrent p150a

Make full-HD pictures with [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (the base model, not a
distilled or turbo version) on two Tenstorrent p150a cards connected by their direct links, or on one card. 50 steps,
guidance 4, 8-bit (BF8) weights, made at 1920x1088 and cropped to 1920x1080. Text encoder, diffusion transformer and VAE
all run on the cards.

    cd qwen-image-2512-p150x2
    export MODEL=/path/to/qwen-image-2512 CACHE=/path/to/cache   # the weights (see below) and a cache folder
    ./run.sh setup                                               # once: code from the image + our patches, checked
    ./run.sh gen --model $MODEL --cache $CACHE                   # the five test pictures, one model over both cards

On our box the final setup makes **53.6 pictures an hour** with one model split over both cards (67 s per picture) and
**55.8 an hour** with one whole model per card (two pictures at a time, 128 s each). Every number below comes from the
files in `results/`, and `results/table.py` prints the tables from them.

Episode 3 of the [Exit the Moat](https://www.youtube.com/@ExitTheMoat) channel. Not affiliated with or endorsed by
Tenstorrent or Alibaba Cloud (Qwen).

## What you need

- **Cards:** two Tenstorrent p150a with **two QSFP-DD cables** between them (4 direct links, the `P150_X2` setup; the
  tensor split uses all 4), or one p150a for `--mode card`. Power each card with its own native 12V-2x6 cable from the
  PSU; a PCIe-to-12V-2x6 adapter made our PSU shut off at power-on.
- **Host:** Linux, x86-64 (we used 8 cores / 16 threads, 62 GB RAM, PCIe Gen3 x16 + x4). `podman` run as root (the card
  and hugepage devices are passed through), `git`, `python3` (standard library only, for the summaries).
- **Driver and firmware** (as measured): [tt-kmd](https://github.com/tenstorrent/tt-kmd) **2.11.0**, firmware bundle
  ([tt-firmware](https://github.com/tenstorrent/tt-firmware)) **19.15.0**, tt-smi 6.6.0. Other versions may work; we
  did not test them.
- **Hugepages:** 1 GB hugepages, 4 per card, mounted at `/dev/hugepages-1G`. Check with
  `cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` (expect 8 for two cards). The `tenstorrent-tools` package
  ([tt-system-tools](https://github.com/tenstorrent/tt-system-tools)) sets this up.
- **Software:** the public tt-inference-server image (about 15 GB; it has a prebuilt tt-metal/ttnn):

      podman pull ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af

  Its `models/tt_dit` is identical to tt-metal commit
  [`de59f8a658b1`](https://github.com/tenstorrent/tt-metal/commit/de59f8a658b1ceafd230c8266026b1a72bb198d7)
  (git tree `b414bc48a682`). `patches/` (22 commits, all inside `models/tt_dit`) go on top of it.
- **Weights** (about 54 GB, Apache-2.0), the revision we used:

      pip install -U huggingface_hub
      hf download Qwen/Qwen-Image-2512 --revision 25468b98e3276ca6700de15c6628e51b7de54a26 --local-dir $MODEL

- **Disk** for `--cache`: the 8-bit weight cache is written on the first start of each layout: tensor split 29 GB, one
  whole model per card (`copies`, `card`, `guidance`) another 28 GB, plus 1-2.5 GB of compiled kernels. Both together
  came to 59 GB in our test.

## Set up

All commands run from this folder (`cd qwen-image-2512-p150x2`); relative paths such as `out/split` are relative to it.

    export MODEL=/path/to/qwen-image-2512 CACHE=/path/to/cache
    ./run.sh setup        # prints SETUP_OK
    ./run.sh smoke 0 1    # expect SMOKE_OK for each card

`setup` copies `models/tt_dit` out of the image into `./work/tt-metal`, checks that it is the expected base, applies
`patches/` with `git am` and checks the result (git tree `e180d768b98d`). The containers mount that folder over the
image's copy; nothing else in the image changes. If you already have a tt-metal checkout, the same patches apply there:
`git checkout de59f8a658b1 && git am /path/to/patches/*.patch`.

## Start

    ./run.sh gen --model $MODEL --cache $CACHE

makes the five test prompts of `harness/prompts.json` with our seeds into `./out/split/` (`<prompt>-s<seed>.png` at
1920x1088, a `_1920x1080.png` crop, a `.json` with the settings and the time of every step) and prints the time per
picture. Options (`./run.sh --help`):

| option | what |
|---|---|
| `--mode split` | default: one model over both cards (tensor split, the guidance branches in one batch) |
| `--mode copies` | one whole model per card; two pictures at a time from one shared job list |
| `--mode card` | one card alone (`--device N`, default 0) |
| `--mode guidance` | each card holds a whole model and computes one of the two guidance branches |
| `--size 1024` | 1024x1024 instead of full HD |
| `--prompts a,b` | a subset of `harness/prompts.json` (add your own prompts there; `all` = every prompt) |
| `--prompt-file F` | another prompt file in the same format, e.g. `musicvideo/prompts.json` (with `--prompts`) |
| `--seed-offset N` | add N to every prompt's seed |
| `--out`, `--tag` | output folder and sub-folder |
| `--write-bps /dev/sda:80mb` | throttle the container's writes to that disk (we used it; our SATA SSD is fragile). It does not throttle reads, so weight loading is not slowed. Name the physical disk, not a `/dev/mapper` name |

The first start of a setup converts the weights to 8 bits into `--cache` and compiles the kernels; later starts load the
cache. Measured with this package from a clean copy, tensor split, weights read from a SATA SSD, cache on an NVMe SSD,
`--write-bps /dev/sda:80mb`: with a new, empty cache the pipeline was ready after 475 s (29 GB written: transformer 21 GB, text
encoder 7.5 GB, plus 1.5 GB of compiled kernels), and the five pictures were done 1017 s after the start. With the cache
in place the pipeline was ready after 15 s, and one picture took 111 s from the command to the file. `--mode copies` on
a new one-card cache: card 0 builds the cache (ready after 478 s), card 1 starts once card 0 is ready (12 s), so the two
never write the cache at the same time. A second test from the README alone, without `--write-bps` (weights and cache
on NVMe), gave the same: ready after 480 s.

**The first start prints nothing to the terminal for about 8 minutes** while it converts the weights and compiles,
and nothing per picture until all five are done (about 17 min in total on a cold cache). The progress is in the log:
`tail -f out/logs/split.log` (`out/logs/<tag>.log`; two copies: `<tag>_card0.log` and `<tag>_card1.log`). The log's
timestamps are UTC (the container's clock); the lines `run.sh` prints are in your local time.

The log has many warnings that are expected and harmless: "Failed to discover available ethernet links; falling back
to 1 link", "Unknown motherboard", and "no known best blocking" for conv3d and matmul shapes. The tensor split still
uses its card-to-card links (the speed per picture shows it; one card alone is about half as fast), and the blocking
warnings only mean ttnn picks a default program configuration for that shape.

Single runs are slower per picture than the counted queue: each new prompt length and the first VAE call compile
programs on first use (first picture 213 s, then 69-82 s, versus 67 s in a warmed-up queue). The queue does a short
warm-up per prompt first; its numbers are the ones in the tables.

`./run.sh queue ...` takes the same options and runs our counted queue (below): a warm-up, then 20 pictures, and appends
one summary line to `out/queue.jsonl`.

**Check the output.** On our box every start produced byte-identical pictures for the same prompt, seed and setup
(across starts, processes and days). `reference/sha256.txt` holds the hashes of our final pictures:

    ./run.sh verify out/split split          # expect IDENTICAL 5/5
    ./run.sh verify out/card card            # one card, two copies and guidance per card make the same pictures:
    ./run.sh verify out/copies card
    ./run.sh verify out/guidance card
    ./run.sh verify out/split-1024 split-1024
    ./run.sh verify out/copies-1024 card-1024

We have not tried a second machine; a different image, weight revision or firmware may change the bytes. If the
hashes differ, compare the pictures by eye or with `check/compare.py` before assuming something broke.

## What we measured

Measured on 11 and 12 November 2026: tt-kmd 2.11.0, firmware 19.15.0, image `0.20.0-de59f8a-03fa3af`, patches up to
0022, one host with 8 cores / 16 threads, PCIe Gen3 x16 (card 0) and x4 (card 1), nothing else on the cards (other,
CPU-only services ran on the host).

**Method (counted queue).** Five prompts (`keyart`, `text`, `portrait`, `detail`, `moat`) x 4 seeds (the prompt's seed
+ 0..3) = 20 pictures, after a 2-step warm-up per prompt in the same process. Wall clock from the first job's start to
the last picture saved; pictures per hour = 20 x 3600 / wall clock. "s per picture" is the time of one job inside the
queue (two copies: two jobs run at the same time). Each start is a new container. The second one-card start ran 10
pictures. Everything on the cards: text encoder, 50 denoising steps, VAE.

| setup | size | pictures/h, start 1 / start 2 | pictures per queue | s per picture (one job) | encoder / denoising / VAE s | s per step |
|---|---|---|---|---|---|---|
| tensor split (one model over both cards) | full HD | **53.68 / 53.54** | 20 / 20 | 67.06 | 0.1 / 65.25 / 0.53 | 1.305 |
| two copies (one whole model per card) | full HD | **55.84 / 55.77** | 20 / 20 | 127.8 | 0.15 / 122.75 / 1.83 | 2.455 |
| guidance per card | full HD | **38.00 / 38.00** | 20 / 20 | 94.74 | 0.14 / 92.56 / 0.85 | 1.851 |
| one card alone | full HD | **29.07 / 29.09** | 20 / 10 | 123.82 | 0.14 / 119.6 / 1.8 | 2.392 |
| tensor split | 1024x1024 | **105.45 / 105.37** | 20 / 20 | 34.08 | 0.1 / 33.1 / 0.26 | 0.662 |
| two copies | 1024x1024 | **109.93 / 110.01** | 20 / 20 | 64.37 | 0.15 / 60.3 / 1.48 | 1.206 |

"Faster pictures or more pictures": the tensor split gives one picture in 67 s; two copies give slightly more per hour
but each picture takes 128 s.

The same queues at the start of the optimisation work (config v2, same method; the four tensor-split starts drifted
between afternoon and evening, cause not found):

| setup | v2 | final |
|---|---|---|
| tensor split | 36.18 / 34.88 / 39.44 / 39.17 | 53.68 / 53.54 |
| two copies | 37.41 / 37.46 | 55.84 / 55.77 |
| guidance per card | 30.70 / 30.68 | 38.00 / 38.00 |
| one card alone | 20.78 / 20.92 | 29.07 / 29.09 |

**Energy** (tensor split, first counted start; passive hwmon log at ~1 Hz, `results/power_q9_b_r1.csv`): cards 118.3 +
115.2 W mean, **4.35 Wh per picture on the cards**, **11.57 Wh per picture at the PSU** (620.9 W mean; the PSU figure
is the whole box including the host and its other services). Wh per picture = mean power x queue wall clock / 20. The log
came from `check/powerlog.sh out.csv` (start it before the queue, `kill` it after; the PSU columns need a PSU that
reports to hwmon, ours is a Corsair HX1500i).

**Where the time goes** (tensor split, one forward of the transformer on card 0, tracy device profiler,
`results/profile_tensor_split_2cards_v3a8flq.json`): the device is busy 0.598 s per forward plus 0.054 s not on the
device; matmuls 46 % of the busy time (30 % with their card-to-card gathers fused in, 16 % plain), attention 22 %, the
rest small memory-bound ops (head split and concat, norms, RoPE, reduce-scatter, modulation, casts). Further gains need attention kernel work
or a different parallel split.

## Quality

The cards run the same algorithm as the stock diffusers pipeline (guidance with diffusers' norm rescale, each guidance
branch with its real prompt length; patch 0002, `--true-cfg --trim-prompt`). To check it, we made the five test pictures
on an RTX 4080 SUPER with stock diffusers, bf16, **with the same seed and the same start noise** (`reference/rtx_ref.py`:
`torch.manual_seed(seed)` + `randn` on the CPU, exactly as `harness/gen.py` does) and compared picture by picture
(`check/compare.py`: PSNR, SSIM, CLIP image-image cosine):

| prompt | PSNR dB | SSIM | CLIP |
|---|---|---|---|
| detail-s13 | 16.56 | 0.5898 | 0.9597 |
| keyart-s2 | 19.09 | 0.6904 | 0.8978 |
| moat-s2 | 23.75 | 0.822 | 0.9517 |
| portrait-s12 | 22.59 | 0.8963 | 0.9555 |
| text-s11 | 13.46 | 0.614 | 0.9043 |
| **mean** | **19.09** | **0.7225** | **0.9338** |

(`results/compare_set_v3p_vs_rtx_base50.json`, final setup, tensor split.) How to read it: 50 denoising steps amplify
tiny numerical differences. Rewriting only the Euler update (scalar vs tensor step size) already moved one picture by
28.7 dB, so 25-29 dB is the noise floor of "the same picture"; below it, pictures show the same layout and subject with
different detail. On the way the mean went from 22.22 dB (v2, `compare_set_v2a_vs_rtx.json`) and 22.37 dB (v3,
`compare_set_v3c_vs_rtx_base50.json`) to 19.24 dB with LoFi matmuls (`compare_set_v3n_vs_rtx_base50.json`) and 19.09 dB
final. Each of those steps was accepted by eye on a side-by-side page against the RTX pictures: same composition, people
and lettering. One visible change: in the `text` picture the 8-bit attention added a third, cut-off line of chalk
scribble under the two correct lines. The `keyart` prompt (the channel's banner prompt) misspells the tagline on some
seeds: 5 of the 19 seeds we made on the final setup (`results/runs_final.jsonl`, tag `final_keyart_seeds`, checked by
eye). The test set uses fixed seeds; we did not seed-search.

`results/pcc.jsonl` holds the development check of one transformer step against the CPU diffusers model (PCC 0.99985 or
better, bf16 and bf8, early configuration).

To make the RTX set yourself (any CUDA GPU; 16 GB works with the block offload the script uses; we had diffusers
0.42.1, transformers 5.22.2, torch 2.11.0+cu128, accelerate 1.15.0):

    python reference/rtx_ref.py --tag base50 --prompts keyart,text,portrait,detail,moat --out rtx
    python check/compare.py out/split rtx/base50 split_vs_rtx

The RTX timings are not a speed comparison: the 16 GB card streams the 41 GB transformer from system RAM.

## The final configuration

`run.sh` holds every flag in one place (`COMMON`, `FUSE_SPLIT`, `FUSE_CARD`, `ENV_ALL`). Without these flags the
patched `tt_dit` runs its default path unchanged; every change in `patches/` is behind a switch.

| switch | what it does |
|---|---|
| `--weights bf8` | 8-bit (BF8) transformer weights, converted into the cache on the first start |
| `--true-cfg --trim-prompt` | diffusers' guidance norm rescale; each guidance branch uses only its real prompt tokens |
| `--encoder device --vae device` | text encoder and VAE on the cards (`--static-load`: kept there in the two-card splits) |
| `TT_DIT_QWEN25VL_CENTER_K=1` | text encoder: subtract the mean key in attention (exact in math; fixes a precision loss in the first layer) |
| `TT_DIT_QWEN25VL_WEIGHTS=bf8` | 8-bit text encoder weights |
| `psdpa` + `EP3_PSDPA_CHUNKS=320x512` | plain streaming attention over [picture; prompt] instead of the joint attention op |
| `modcache` + `EP3_MODCACHE=step` | the timestep modulation once per step (pictures byte-identical to without) |
| `addcmul`, `rope`, `attn_out`, `qkv`, `ff1`, `up` | fused block ops: gated residual as one op, the fused RoPE op, gather + output projection + gated add in one op, card-to-card gathers fused into the qkv and ff1 matmuls (`up`: output projection weights in bf16 on the card, needed by the fused op; tensor split only) |
| `gelufast` | the stock fast (lookup-table) GELU |
| `--mm-fid lofi` | LoFi math fidelity on every Linear matmul (attention stays HiFi2) |
| `a8n`, `a8att`, `a8o` | 8-bit inputs: norm outputs into the big matmuls and gathers, q/k/v into attention, attention output into the output projection |
| `--links 4 --norm-links 3` | tensor split: gathers on all 4 links, the fused norms on 3 |
| `EP3_VAE_EVICTS_ENCODER=1` | whole model on one card: the VAE swaps with the text encoder to fit in memory |

Residual stream, norms, softmax, accumulation and the guidance combine stay in higher precision.

**Levers kept** (each measured on counted queues or image sets, and approved by eye): streaming attention (forward
0.860 -> 0.787 s), modulation once per step (byte-identical), 8-bit norm outputs + 8-bit text encoder, fast GELU, LoFi
matmuls, 8-bit q/k/v and attention output (step -0.095 s).

**Tried and not kept:** fused per-head q/k norm + RoPE (pictures diverge), norm fidelity, two attention calls instead of
one concatenated query (slower), fused ff2 + reduce-scatter + gated add (needs a ring topology; the ring hung the cards
three times), a cheaper GELU polynomial (gain gone in queues), 8-bit q/k/v before RoPE (the RoPE op takes BF16 only),
8-bit output projection weights unfused (slower), 8-bit attention output alone (no gain), attention chunk sweeps and
exp/math approximations, RoPE fidelity, matmul blocking overrides, larger attention chunks with 8-bit attention (slower
or out of L1), precomputing the modulation for all steps (no gain), and **tracing the whole step**: the first picture of
a process is right, every later one is noise (cause not found; `patches/0003` and `0011` keep the code, off by default).

## Optional: the music-video pictures

The three music videos at the end of the episode use 122 pictures drawn on the two cards with the final setup above
(tensor split, full HD). `musicvideo/prompts.json` holds their prompts, negative prompts and seeds (`s1-..`, `s2-..`,
`s3-..` = versions 1-3 of the song). To redraw one or two:

    ./run.sh gen --model $MODEL --cache $CACHE --prompt-file musicvideo/prompts.json --prompts s1-01,s3-42 --tag musicvideo
    ./run.sh verify out/musicvideo musicvideo     # expect IDENTICAL 2/2 (we keep hashes for these two only)

The pictures, the song and the animation are not part of this repository.

## Limits worth knowing

- **Our target was 120 pictures an hour at full HD. Not reached:** best 55.8 (two copies). One picture is about
  1.63e16 floating-point operations; at the matmul rates we measured on these cards and under the board power limit,
  the realistic ceiling is well below 120.
- **The board power limit.** Each chip has `tdp_limit` 150 W and `tdc_limit` 200 A (`results/power_limits_stock.json`).
  In sustained picture work the chips run at a median clock of about 1160-1190 MHz (of 1350), at 150-160 W and 190-215 A,
  card 0 at 83-86 °C in our case. The limits sit in the card's firmware table; tt-smi cannot set them and we did not
  change them (that would be a firmware change). Short benchmarks run nearer full clock, so they overstate compute-only
  gains.
- The tensor split needs both cables (4 links). Sizes must be multiples of 16; we measured only 1920x1088 and 1024x1024.
- Byte-identical pictures are per box: identical across all our starts; untested on another machine.
- One whole model on one card is tight in memory (the VAE and the text encoder take turns).
- The patches also contain switches we used for development only (traced steps, q/k/v dump, bf4 weights, the
  Qwen-Image-Edit-2511 timestep path). They are off unless you set them and none was measured for this package.

## Layout

| path | what |
|---|---|
| `run.sh` | setup / smoke / gen / queue / verify; every mount, flag and environment variable is in this one script |
| `patches/` | 22 commits on tt-metal `de59f8a658b1` `models/tt_dit` (`git am`) |
| `harness/gen.py` | makes pictures (single runs and queues), writes `.png`, `.json` and `runs.jsonl` |
| `harness/ep3_levers.py`, `harness/common.py` | the switches applied to the built model; mesh helpers |
| `harness/prompts.json` | the five test prompts and their seeds |
| `harness/qsum.py` | queue summary (pictures per hour from the run lines) |
| `musicvideo/prompts.json` | prompts and seeds of the music-video pictures (optional, `--prompt-file`) |
| `check/smoke.py` | opens a card and checks one matmul |
| `check/powerlog.sh` | passive power log from hwmon at ~1 Hz (cards, and the PSU if it reports) |
| `check/compare.py`, `check/sheet.py` | PSNR / SSIM / CLIP between two picture sets; a contact sheet |
| `reference/rtx_ref.py` | the diffusers reference with the same start noise |
| `reference/sha256.txt` | hashes of our final pictures, per setup (and of music-video pictures s1-01 and s3-42) |
| `results/queue.jsonl` | every counted queue of the work (final configs end in `_v3a8flq`; earlier ones are the history) |
| `results/runs_final.jsonl` | one line per final picture (queues, the comparison set, the 1024 detail and keyart seeds) |
| `results/power_*`, `results/profile_*`, `results/compare_*`, `results/pcc.jsonl` | power log and summary, card limits, profiles, quality comparisons, step PCC |
| `results/table.py` | prints the tables in this README from those files |

## Licences

- This package's code: Apache-2.0 (see `LICENSE` at the top of this repository). `patches/` modify Tenstorrent tt-metal
  files (Apache-2.0).
- Qwen-Image-2512: Apache-2.0 (Alibaba Cloud / Qwen team; see the model card). No weights are part of this package; the
  8-bit conversion happens in your cache on your machine.
