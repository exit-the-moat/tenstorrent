# Wan 2.2 A14B on 2 Tenstorrent p150a

Make 5-second video clips with [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (the 14B
text-to-video model with its two experts, 40 steps, guidance; plus image-to-video and two distilled 4-step variants) on
two Tenstorrent p150a cards connected by their direct links. Text encoder (umT5), diffusion transformer and VAE all run
on the cards. Tenstorrent's `models/tt_dit` has a Wan 2.2 pipeline for 4, 8 and 32 chips; it has no preset for two.
The 13 patches here make it run on two, add the speed levers, and fix what we found on the way.

    cd wan22-a14b-p150x2
    export MODELS=/path/to/models CACHE=/path/to/cache   # the downloads (see "Weights") and a cache folder
    ./run.sh setup                                       # once: code from the image + our patches, checked
    ./run.sh t2v                                         # the counted clip twice (480p, best preset)

On our box the best preset makes a 480p clip (832x480, 81 frames = 5 s at 16 fps) in **503 s** and a 720p clip in
**1,826 s**; the distilled 4-step Lightning model does the same 480p clip in **31 s**. Every number below comes from
the files in `results/`.

Episode 4 of the [Exit the Moat](https://www.youtube.com/@ExitTheMoat) channel (with LTX-2.5 in
[`../ltx25-p150x2/`](../ltx25-p150x2)). Not affiliated with or endorsed by Tenstorrent, Alibaba (Wan) or LightX2V.
**Every clip this code makes is AI-generated; label it as such wherever you publish it.**

## What you need

- **Cards:** two Tenstorrent p150a with **two QSFP-DD cables** between them (4 direct links, the `P150_X2` setup).
  Power each card with its own native 12V-2x6 cable from the PSU; a PCIe-to-12V-2x6 adapter made our PSU shut off at
  power-on.
- **Host:** Linux, x86-64. Ours: 8 cores / 16 threads, **62 GB RAM**, PCIe Gen3 (card 0 x16, card 1 x4), models and
  caches on an NVMe SSD. `podman` run as root, `git`, `python3`.
- **Driver and firmware** (as measured): tt-kmd **2.11.0**, firmware bundle **19.15.0**. 1 GB hugepages, 4 per card
  (`cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` should say 8).
- **Software:** the public tt-inference-server image (prebuilt tt-metal/ttnn, nothing to compile):

      podman pull ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af

  Its `models/tt_dit` is identical to tt-metal commit
  [`de59f8a658b1`](https://github.com/tenstorrent/tt-metal/commit/de59f8a658b1ceafd230c8266026b1a72bb198d7) (git tree
  `b414bc48a682`, the same base as Episode 3). `patches/` (13 commits, all inside `models/tt_dit`) go on top of it;
  `setup` checks the result (tree `678329539390`). With your own checkout: `git checkout de59f8a658b1 && git am --keep-cr
  patches/*.patch`, or `TT_DIT_SRC=/path/to/tt-metal ./run.sh setup`.

## Weights

Put each download in its own folder under `$MODELS` (the names matter, `run.sh` mounts `$MODELS` as `/models`):

    pip install -U huggingface_hub
    hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 --exclude "assets/*" --local-dir $MODELS/wan22-a14b
    # only for i2v / distill / svi:
    hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --revision 596658fd9ca6b7b71d5057529bbf319ecbc61d74 --exclude "assets/*" --local-dir $MODELS/wan22-i2v-a14b
    hf download lightx2v/Wan2.2-Distill-Models --revision 52699089e1dac804cbedd2400d0d2aea969577de --local-dir $MODELS/lightx2v \
        wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors config.json
    # only for lightning:
    hf download lightx2v/Wan2.2-Lightning --revision 18bccf8884ec0a078eed79785eb4ef13ea16ce1e --local-dir $MODELS/lightning \
        --include "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/*"
    # only for svi:
    hf download vita-video-gen/svi-model --revision f9bdd50c57e4c28f0d088b6bad8afa3fdc3bc1ac --local-dir $MODELS/svi/svi --include "version-2.0/*"

The T2V and I2V checkpoints are 126 GB each, because both 14B experts are stored in fp32. The first start of a setup
converts them into `$CACHE` (bf8: about 32 GB for both experts, 11 GB for umT5). After that the fp32 transformer files
are only read again for a new cache (another dtype or split); we deleted ours after the caches were built.

## Set up and run

    ./run.sh setup             # prints SETUP_OK (also installs imageio-ffmpeg into ./work/pylib for the mp4 export)
    ./run.sh smoke 0 1         # expect SMOKE_OK for each card
    ./run.sh t2v                              # 480p, best preset, the counted job (two boxing cats, seed 42, twice)
    ./run.sh t2v --res 720p                   # 720p: best preset + memory plan + expert swap
    ./run.sh t2v --preset stock               # stock tt_dit (+ lazy loading), bf16, expert swap, 2 links
    ./run.sh t2v --jobs harness/prompts8.json # the 8-prompt harness (motion, face, text, crowd, pan, ...)

Each job writes `out/<run>/<name>_s<seed>.mp4` (and `.npy` frames) and one line in `out/runs.jsonl` with the times of
encoder, denoising (every step) and VAE. `python3 harness/scoreboard.py` (inside the container: `./run.sh py
scoreboard.py`) turns `runs.jsonl` into `results.csv`. A run starts with an untimed 2-step warm-up; the first start of
a setup also builds the cache (480p bf16 stock: 276 s cache build + 188 s warm-up in our first run).

Image-to-video and the distilled models (jobs are json lists of `{name, seed, prompt, image}`; copy the start image
into `$OUT` and give it as `/out/<file>`; crop it to 832:480 yourself or let the runner crop the centre):

    ./run.sh i2v --jobs my_i2v.json           # Wan2.2-I2V-A14B, 40 steps, guidance 3.5 / 3.5
    ./run.sh distill --jobs my_i2v.json       # lightx2v 4-step distilled I2V, reference sampler, no CFG
    # SVI 2.0 Pro: 4 chained 5 s clips = 312 frames (19.5 s), the SVI LoRA merged on the host into the distilled weights
    S=/models/svi/svi/version-2.0/SVI_Wan2.2-I2V-A14B
    EP4ENV="EP4_MERGE_LORA_HIGH=${S}_high_noise_lora_v2.0_pro.safetensors EP4_MERGE_LORA_LOW=${S}_low_noise_lora_v2.0_pro.safetensors EP4_MERGE_LORA_TAG=svi20pro" \
        ./run.sh svi --jobs my_i2v.json --clips 4
    # Lightning T2V: merge the LoRA into both experts on the host (fp32, ~14 GB RAM, needs the fp32 T2V download), then run
    ./run.sh merge-lora transformer   lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors lightning11
    ./run.sh merge-lora transformer_2 lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors  lightning11
    ./run.sh lightning --merged lightning11   # 480p counted job; --res 720p for 720p

**What we checked and what we did not:** the patches and `setup` were checked byte for byte (the patched tree equals
the one every counted run used). `run.sh` wraps the same runners, presets and flags as our runs (they are listed in
each line of `results/runs.jsonl`), but the wrapper itself was written for this release and has **not** been run
end to end on the cards from this folder. The runs in `results/` came from the same Python files with our own
launch scripts.

## What we measured

17-18 November 2026, tt-kmd 2.11.0, firmware 19.15.0, image `0.20.0-de59f8a-03fa3af`. Counted = one process per
config, an untimed warm-up, then the same clip twice (cats, seed 42), host otherwise idle. Seconds per clip include the
text encoder, 40 denoising steps (each = two transformer passes, with and without the prompt) and the VAE.
Same-config repeats gave **bit-identical** clips (sha256), so a hash is a cheap regression test.

**480p lever ladder** (closed PC case, `results/results.csv`):

| config | run 1 / run 2 (s) | vs stock | s per step |
|---|---|---|---|
| first 2-card clip: stock tt_dit + lazy loading, bf16, expert swap, 2 links (`first480`) | 661.2 | | |
| L0 stock (same, counted pair) | 667.0 / 670.2 | | 16.52 |
| L1 + bf8 weights, both experts stay on the cards | 615.9 / 618.8 | -7.7 % | 15.28 |
| L2 + attention chunks 512x256 | 587.5 / 588.9 | -12.0 % | 14.55 |
| L3 + 4 links instead of 2 (clips bit-identical to L2) | 566.1 / 567.6 | -15.2 % | 14.01 |
| **L4** + bf8 attention inputs, chunks 512x512 (**best**) | 525.2 / 529.0 | **-21.2 %** | 13.02 |

**After moving the server into an open frame** (better cooling; same code, `K*` runs in `results.csv`):

| config | closed case | open frame |
|---|---|---|
| 480p stock | 667.0 / 670.2 | 642.7 / 647.1 (-3.5 %) |
| 480p best (L4) | 525.2 / 529.0 | **503.1 / 502.6** (-4.6 %) |
| 720p best + memory plan | 1,832.4 / 1,834.9 | **1,825.9** (-0.4 %) |

Our target was -30 % against the first 2-card clip (<= 463 s). **Missed: -23.9 %** (503 s vs 661 s). The cards were
clock-bound, not waiting on the host: under this load card 0 ran at ~974-1,066 MHz and card 1 at ~1,111-1,149 MHz of
1,350 (power log), and the slower card sets the pace at every gather. At 720p both cards sat at ~130 W each in both
cases: the limit there is the cards' power cap (firmware), not heat. Tenstorrent publishes 466 s for the same 480p job
on a 4-chip QuietBox (`models/Wan2_2.md`, published, not our run) and no 720p number for a 4-chip box.

**720p** (1280x720, stock code does not fit on two cards; memory plan needed, see below): stock + memory plan
2,356.3 / 2,341.1 s, best + memory plan 1,832.4 / 1,834.9 s (-21.9 %).

**Image-to-video and distilled models** (480p, 81 frames, open frame unless noted; `results.csv`):

| model | s per clip | note |
|---|---|---|
| Wan2.2-I2V-A14B, 40 steps, best + memory plan + expert swap (closed case) | 524.6-529.0 | encoder 2.3 s (reload after eviction), VAE 6.3 s |
| lightx2v distilled I2V, 4 steps, reference sampler (counted pair `D3r`) | **52.2 / 51.1** | 10x faster than the 40-step I2V |
| Wan2.2-Lightning T2V, 4 steps, LoRA host-merged, bf8 (counted pair `G6`) | **30.7 / 31.0** | 16.4x faster than the 40-step best (503 s) |
| Lightning T2V at 720p (`G7`) | 111.2 | 16.4x faster than 1,826 s |
| SVI 2.0 Pro on the distilled I2V, 4 clips = 19.5 s (`V2`) | 166.7 / 168.2 | 52-53 s for the first clip, 38.2 s per further clip |

The distilled models are **different models** (trained to imitate the 40-step one in 4 steps, guidance baked in), not
a faster version of the same computation; judge their clips by eye.

**Where a 480p step goes** (L3, tracy device profile of one transformer pass on card 0, rescaled to the real clock):
attention (SDPA, 40 self- and 40 cross-attention calls) 50 %, matmuls 35 %, fused norms 7.5 %, card-to-card
collectives 3 %, rest 4 %. The device span equals the synced wall time: nothing waits on the host, which is why tracing
the step gave nothing. The text encoder is negligible: 0.134 s warm per prompt (with the negative prompt, 512 tokens,
`results/enc_time.jsonl`).

**Energy** (passive hwmon log every 5 s: both cards and the PSU output = the whole PC including this server's other
always-on services; `results/power_summary*.csv`, `harness/powersum.py`, 0.35 EUR/kWh):

| | cards (both) | whole PC | Wh per clip (whole PC) |
|---|---|---|---|
| idle | 38 W | 181 W (closed case) / 164 W (open frame) | |
| 480p best | 239-260 W | 603-640 W | 84.9-91.8 (~3 ct) |
| 720p best + memory plan | 254-257 W | 617-620 W | 313.8-315.8 (~11 ct) |
| 720p stock + memory plan | 248-251 W | 619-623 W | 405.4 (~14 ct) |
| Lightning 480p / 720p | | | 4.7-5.8 / 19.4 |

**For comparison, one rented NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)**, stock diffusers 0.38.0 bf16 with
PyTorch SDPA (no FlashAttention, no FP8, no torch.compile), the same checkpoint revision, scheduler, guidance, negative
prompt and start noise (`reference/ref_wan22_gpu.py`, `results/rtx_pro_6000/`): 480p **554.5 / 558.6 s**, 720p
**1,858.3 / 1,876.7 s** (at 720p the experts are moved to the CPU before the fp32 VAE decode; ~15 s inside the total),
harness prompts 533.3-559.5 s. GPU board ~588 W under load, ~91 Wh per 480p clip for the GPU alone (our two cards:
~245 W, ~34 Wh per clip with the best preset; neither is wall power). So stock against stock the GPU is ~14 % faster;
our best preset (bf8, a precision change) is ~10 % faster than its stock bf16 at 480p and on par at 720p. A tuned GPU
path would be faster. Same prompt and seed give similar scenes but different details on the two implementations
(diffusers vs tt_dit differ numerically; not investigated).

## Correctness gates

1. **PCC against a CPU fp32 reference** (stock diffusers `WanTransformer3DModel`, block-streamed from the safetensors
   so it fits in RAM; `check/cpu_ref.py`): the card dumps its exact inputs at chosen steps (`check/pcc_card.py`, 480p,
   17 frames, seed 7), the CPU runs one transformer pass on them, `check/pcc_cmp.py` compares (`results/pcc.jsonl`).
   Gate: PCC >= 0.99 for the conditional and unconditional pass.
   - T2V stock bf16, steps 0 / 10 / 30 / 39: 0.999852 / 0.999922 / 0.999496 / 0.998962; best (bf8 + bf8 attention)
     at steps 0 / 10 / 30: 0.999893 / 0.999905 / 0.999500.
   - I2V (best + memory plan), steps 0 / 10 / 30 / 39: >= 0.999243 (cond).
   - Distilled I2V, reference sampler, all 4 steps, both experts: 0.999803-0.999854.
   - **Lightning T2V: fails the gate on one step.** Steps 0 / 1 / 3 pass (0.99958 / 0.99692 / 0.99311); the first
     low-noise step (step 2, sigma 0.83) reaches only 0.97143 (0.98476 with bf16 attention). See "Known issues".
2. **Two counted runs** per config, and **by eye** against the stock clip with the same seed (`harness/sheet.py`
   side-by-sides). The bf8 attention of L4 changes per-seed details: in our test the white boxing cat got light patches
   on its coat (20.2 dB PSNR vs stock); composition, motion and timing stay. bf8 weights alone: 23.8 dB, no visible
   change. The channel owner watched both side by side and accepted L4.
3. Bit-identity for levers that should not change numerics (4 links, tracing, the memory plan): sha256 of the clip.

## The presets (`run.sh`)

| flag | what |
|---|---|
| `EP4_DIT_WEIGHTS=bf8` | Linear weights converted and cached as bf8; the self-attention output projection stays bf16 (its fused kernel needs it). Both experts fit next to umT5 at 480p, so no swap (`--dynamic-load 0`) |
| `EP4_SDPA_CHUNKS=512x512`, `EP4_SDPA_BF8=1` | attention chunk sizes; bf8 q/k/v into the self-attention (cast after the norm, output cast back to bf16) |
| `--links 4` | all-gathers / reduce-scatters on the 4 links (ignore the log line "falling back to 1 link": it comes from another helper; 4 links measurably sped up the gathers and gave identical clips) |
| `EP4_ENC_EVICT=1`, `EP4_VAE_EXCL=1` | the memory plan: free umT5 after encoding (reload from the cache, ~2.2 s), never keep the VAE and the transformers on the cards together. Needed at 720p and for every I2V run, with `--dynamic-load 1` |
| `EP4_DISTILL_SAMPLER=ref` | 4-step models: LightX2V's reference schedule (sigmas 1.0 / 0.9375 / 0.8333 / 0.625, flow Euler, 2 high-noise + 2 low-noise steps, boundary 0.875). tt_dit's experimental default never reaches the low-noise expert at 4 steps (timesteps 999 / 973 / 923 / 800): clips pop from frame 0 to 1 and fast motion smears |
| `EP4_CKPT`, `EP4_MERGE_LORA_*` | a host-merged checkpoint / a LoRA merged in fp32 before the cache is built |
| no flag | stock tt_dit behaviour (+ lazy loading): every patch is behind a switch or is a pure memory/port fix |

The 13 patches: lazy bf16 torch weights (the stock code loads both fp32 experts and umT5 into RAM at start-up, ~137 GB
on a 62 GB host), the bf8 weight path, the bf8 attention inputs, a tuned-matmul-table hook, the two memory switches,
CFG batching (blocked, see below), the distilled pipeline's cache key, the host LoRA merge, per-expert weight dtypes,
and three precision switches used only to diagnose the Lightning gap (`EP4_SDPA_HIFI4`, `EP4_MM_HIFI4`,
`EP4_XATTN_MANUAL`, `EP4_SATTN_MANUAL`, `EP4_FF_HIFI4`; all off).

## Tried and not kept

- **Tracing the denoising step:** bit-exact, no speed change (L3 563.0 / 565.9 vs 566.1 / 567.6 s; L4 528.6 / 521.8 vs
  525.2 / 529.0 s). The cards are busy the whole pass.
- **Matmul blocking sweep:** the defaults already reach 262-267 TFLOPS; the best qkv blocking was 0.6 % faster.
- **LoFi math:** too inaccurate (PCC 0.82-0.94 in the attention micro-benchmark, `results/sdpa_bench.jsonl`).
- **Guidance passes as one batch-2 forward** (`EP4_CFG_BATCH`): blocked, `nlp_create_qkv_heads` takes batch 1 only.
  It would save ops, not FLOPs, on compute-bound cards.
- **Sequence split or one guidance branch per card** (tp 1): the fused distributed LayerNorm at the full 5120 width needs
  1,631,108 B of circular buffers per core, L1 has 1,572,864 B. Every Blackhole preset in tt_dit uses tp >= 2.
- **Both bf8 experts resident at 720p or with I2V:** warm-up passes, the first clip runs out of card memory; the experts
  are swapped instead (cheap: the bf8 cache reloads from the page cache).
- **tt_dit's runtime LoRA fuse** (adding the LoRA on the cards into weights that are already bf16/bf8): the Lightning
  deltas are about one rounding step of a weight, so most of them are rounded away (only 45-63 % of the LoRA's effect
  survives; clips drift away from the prompt). Fix: merge on the host in fp32 before quantising (patch 0010; the
  streaming `harness/merge_lora_stream.py` needs ~14 GB RAM; merging a whole 57 GB expert in memory got killed).

## Known issues (said plainly)

- **Two card resets** during the work: one attention micro-benchmark configuration (720p shape, q128/k512, LoFi) hung
  card 0; killed containers once left the cards unopenable. Both times a reset (`tt-smi -r 0 1`) and a smoke test
  fixed it. Never kill a container mid-run if you can avoid it.
- **Lightning step 2 PCC 0.971.** Not a mapping bug (all 400 LoRA pairs map, CPU host-merge simulation 0.99985) and
  not inherent to bf16 (a bf16-activation CPU reference stays at 0.9987, `results/pcc_bf16act.jsonl`). The biggest
  card-side error is the fused SDPA kernel (it keeps intermediates in bf16: 1.4-4.7 % local relative error on these
  shapes vs 0.25 % for matmul + fp32 softmax); replacing it lifts step 2 only to 0.980. Card magnitudes are 1-5 %
  smaller than the CPU's from block 0 on. Cause not found. The base model passes on the same inputs.
- **Text in the start frame does not survive camera motion** (I2V): our key art's wordmark lost letters during a
  push-in. The 40-step T2V harness got a licence plate wrong; "OPEN LATE" on a sign came out right.
- 720p and I2V run untraced (the memory plan moves weights; a trace would go stale).
- Sizes: 480p = 832x480, 720p = 1280x720, 81 frames. Other sizes were not measured.

## Layout

| path | what |
|---|---|
| `run.sh` | setup / smoke / t2v / i2v / distill / svi / merge-lora / lightning / py; every mount, flag and preset is in this script |
| `patches/` | 13 commits on tt-metal `de59f8a658b1` `models/tt_dit` (`git am --keep-cr`) |
| `harness/gen*.py` | the runners (T2V, I2V, distilled I2V, Lightning, SVI): job list in, clips + `runs.jsonl` lines out |
| `harness/ep4_distill_sched.py`, `merge_lora_stream.py` | the 4-step reference sampler; the streaming fp32 LoRA merge |
| `harness/prompts8.json`, `jobs_rep2.json` | the 8 test prompts with seeds; the counted job (cats, seed 42, twice) |
| `harness/scoreboard.py`, `powersum.py`, `sheet.py`, `enc_time.py`, `step_time.py`, `sdpa_bench.py` | results.csv from runs.jsonl; energy per clip; contact sheets and same-seed PSNR; encoder, step and attention timing |
| `check/cpu_ref.py`, `cpu_ref_bf16act.py`, `pcc_*.py` | the CPU fp32 / bf16-activation references and the per-step PCC tools |
| `check/powerlog.sh`, `prof_*.py`, `lora_keycheck.py`, `smoke.py` | power log, device profile tools, LoRA key check, card smoke test |
| `reference/` | the RTX PRO 6000 script (`ref_wan22_gpu.py`), the pod set-up and entrypoint we used on the rented GPU |
| `results/` | `runs.jsonl` (every run, one line each), `results.csv` (generated from it), `pcc.jsonl`, `pcc_bf16act.jsonl`, `power_summary*.csv`, `sdpa_bench.jsonl`, `step_time.jsonl`, `enc_time.jsonl`, `mm_sweep.jsonl`, `compare.jsonl`, `rtx_pro_6000/` |

The result files are copies of our run files. Only server paths inside them were rewritten to this package's container
layout (`/models`, `/cache`, `/out`, `/work`); no value was changed.

## Licences

- This package's code: Apache-2.0 (see `LICENSE` at the top of this repository). `patches/` modify Tenstorrent tt-metal
  files (Apache-2.0).
- Wan 2.2 (T2V-A14B, I2V-A14B): Apache-2.0. lightx2v Wan2.2-Distill-Models and Wan2.2-Lightning: Apache-2.0. Stable
  Video Infinity 2.0 Pro (vita-video-gen/svi-model): MIT. No weights are part of this package; the bf8 conversion
  happens in your cache.
- Generated clips are AI output: label them as AI-generated.
