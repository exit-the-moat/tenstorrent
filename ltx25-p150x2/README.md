# LTX-2.5 distilled (video with sound) on 2 Tenstorrent p150a

Make 5-second clips with sound from a prompt (text-to-video) or from a still (image-to-video) with Lightricks'
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) distilled model (22B audio-video transformer, 8 + 3 steps in two
stages, no guidance) on two Tenstorrent p150a cards connected by their direct links. The Gemma 4 12B text encoder, the
transformer, the latent upsampler, the video VAE and the audio decoder all run on the cards. tt-metal's `models/tt_dit`
has a port of the older LTX-2.3 for 8 and 32 chips; the 12 patches here (on top of the Wan 2.2 patches in
[`../wan22-a14b-p150x2/`](../wan22-a14b-p150x2)) add what LTX-2.5 changed, a 2-card setup, the speed levers and long
clips.

    cd ltx25-p150x2
    export MODELS=/path/to/models CACHE=/path/to/cache   # MODELS/ltx25 = the download (see "Weights")
    ./run.sh setup && ./run.sh aux                       # once: code + patches (checked); the small VAE/audio file
    ./run.sh gen --res 720p                              # the three counted prompts at 1280x704, 121 frames + audio

On our box a 5 s clip with sound takes **20.5 s at 720p** (1280x704) and **55 s at 1080p** (1920x1088), with the prompt
already encoded; about 33 s at 720p for a brand-new prompt. Every number below comes from the files in `results/`
or the reports they were written with (named in each table).

Episode 4 of the [Exit the Moat](https://www.youtube.com/@ExitTheMoat) channel. Not affiliated with or endorsed by
Tenstorrent or Lightricks. **Every clip this code makes is machine-generated. The LTX-2.x Community License requires you
to disclose that clearly wherever you publish it** (see "Licences").

## What you need

Same box and software as the Wan 2.2 package: two p150a with two QSFP-DD cables (4 links), tt-kmd 2.11.0, firmware
19.15.0, 1 GB hugepages, `podman` as root, the image
`ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af`
(`models/tt_dit` = tt-metal `de59f8a658b1`). `setup` applies `../wan22-a14b-p150x2/patches` (13) and then `patches/` (12)
and checks the trees (`678329539390`, then `53283b7b96a5`). Host RAM: building the bf8 transformer cache reads the
whole 42 GB bf16 file once; on our 62 GB host that worked with swap and nothing else big running.

## Weights

    hf download Lightricks/LTX-2.5 --revision 49b093a39c19169a8a81721dcc541e07e945232f --local-dir $MODELS/ltx25 \
        diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
        text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
        vae/ltx-2.5-video-vae-conv-bf16.safetensors vae/ltx-2.5-audio-vae-bf16.safetensors \
        latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

The tokenizer and chat template ship inside the Gemma file. `./run.sh aux` writes a 1.8 GB checkpoint with the conv
video VAE, the audio VAE and the vocoder into `$CACHE`, so tt_dit's loaders never open the 42 GB transformer file for
them. The first start writes the bf8 caches (transformer ~12 GB per card share, Gemma 4 ~6 GB).

## Run

    ./run.sh gen --res 720p                    # harness/jobs3.json: motion, face, text (seeds 101-103)
    ./run.sh gen --res 480p --repeat 2         # warm repeats
    ./run.sh gen --res 1080p                   # VAE in 4 width tiles (LTX_VAE_WTILES=4, set by run.sh)
    ./run.sh gen --res 720p --jobs harness/prompts8.json

Each clip lands in `out/gen_<res>/<name>_s<seed>.mp4` (24 fps, stereo 48 kHz) with one line in `out/runs.jsonl` (stage
times). Prompts are encoded on the cards first (Gemma 4), then Gemma is freed and the clips render back to back.

**Image-to-video.** The still is prepared on the host exactly like the official pipeline (H.264 re-compression at
CRF 18, resize, centre crop) with the official code, in a CPU venv (`check/venv.sh` installs `ltx-core` and
`ltx-pipelines` from `github.com/Lightricks/LTX-2` at `f0f2f8607189`, cloned into `$WORK/LTX-2`):

    $WORK/venv/bin/python harness/prep_img.py still.png 1088 1920 $WORK/pre/still_1088x1920
    # job: {"name": "still", "seed": 0, "prompt": "...", "image": "/work/pre/still_1088x1920"}
    ./run.sh gen --res 1080p --jobs my_i2v.json
    # long clips: add "segments": 3, "overlap": 25 to a job (3 chunks, 1 s overlap = 13.0 s), or one single pass:
    ./run.sh gen --res 1080p --frames 249 --long --jobs my_i2v.json   # 10.4 s; job key "single_out": "/out/<file>.mp4"

**What we checked and what we did not:** the patches and `setup` were checked byte for byte (the patched tree equals
the one every counted run used). `run.sh` wraps the same runner and flags as our runs (each line of
`results/runs.jsonl` lists its `LTX_*` variables), but the wrapper was written for this release and has **not** been
run end to end on the cards from this folder.

## What we measured

18-19 November 2026, same box, drivers and image as Wan 2.2. Counted = warm repeats after one untimed warm-up clip in
the same process, host idle, power logged; 121 frames = 5.04 s at 24 fps plus stereo audio. "Render" = with the prompt
embedding already there (`results/runs.jsonl`; stage tables in the worker's report, see below).

**Text-to-video, final preset** (bf8 transformer, borrowed conv3d blockings, 4 links, Gemma 4 bf8 on the cards):

| stage (s) | 480p 832x448 | 720p 1280x704 | 1080p 1920x1088 |
|---|---|---|---|
| text encode per new prompt (Gemma loaded) | 0.23 | 0.23 | 0.23 |
| stage 1, 8 steps (half resolution) | 5.60 / 5.51 | 5.86 / 5.93 | 12.60 / 12.67 |
| latent upsample x2 | 0.08 / 0.08 | 0.16 / 0.16 | 0.35 / 0.34 |
| stage 2, 3 steps | 3.88 / 3.86 | 9.41 / 9.41 | 27.37 / 27.25 |
| VAE decode | 1.20 / 1.20 | 2.79 / 2.80 | 10.97 / 10.97 |
| audio decode | 0.95 / 0.94 | ~0.94 | ~0.95 |
| **render per clip** | **12.40 / 12.28** | **20.51 / 20.58** | **55.03 / 54.84** |
| first working 2-card version (same counting) | 23.9 / 24.1 | 49.8 / 50.1 | 134.0 / 134.4 |
| cards (both) W / whole PC Wh per clip | 126-149 W / 1.4-2.0 | 198-212 W / 3.5-3.6 | 215-244 W / 8.4-8.5 |

Our target was -30 % at 720p against the first working version (<= 35 s). **Reached: -59 %.** Almost all of it came
from one lever: every conv3d of the VAE and the upsampler ran with fallback 1x1x1 blockings on our mesh; borrowing the
blockings Tenstorrent swept for LTX-2.3 on 8 and 32 chips cut the 720p VAE decode from 28.9 to 2.87 s (output PCC 0.99997
against the fallback). 4 links instead of 2: 22.0 / 22.05 -> 20.7 / 20.6 s, output bit-identical.

**A new prompt costs more at 720p and 1080p** (`results/fresh720.json`): Gemma 4 and the 720p transformer activations
do not fit on the cards together, so the cards take turns: reload Gemma from the cache 11.67 s + encode 0.34 s + free +
render 21.0 s = **~33 s for a new prompt at 720p** (~67 s at 1080p). At 480p Gemma stays (fresh prompt ~12.6 s). In a
batch, all prompts are encoded first (one Gemma load 26.5 s, ~0.5 s per prompt), then the clips render back to back.
A fresh process needs ~2.2 min to its first 720p clip (mostly kernel compile).

**Image-to-video** (same preset, every clip its own start image; 121 frames):

| | 480p | 720p | 1080p |
|---|---|---|---|
| image encode on the cards | 0.35 / 0.36 | 0.84 / 0.84 | 1.82 / 1.84 |
| **render per clip** | **13.07 / 13.14** | **23.17 / 23.30** | **59.61 / 59.63** |

**Long clips at 1080p** (I2V):

| | per clip | render s per s of video |
|---|---|---|
| 121 frames (5.0 s), one pass | 59.6 s | 11.9 |
| 3 x 121 frames with 25 frames (1 s) overlap = 313 frames (13.0 s) | 176.6 s | **13.6** |
| **249 frames (10.4 s), one pass** (two-row AdaLN, `--long`) | **178.9 s** warm (32-clip batch median 181.4 s) | 17.2 |
| 377 frames (15.7 s), one pass, only as the first clip of a process | 316.3 / 312.8 s | 19.9 |

Attention grows with the square of the length, so a single pass costs 27-46 % more per second of video than chaining;
it gives one unbroken camera move. The overlap chain (the official method's carry of the last 25 frames) keeps the
motion through the joints (optical-flow direction cosine 1.00 at every joint vs a camera reversal with a 1-frame
carry); decoded overlaps match at a median 34.2 dB.

**For comparison, one rented NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)**, the official `ltx_pipelines`
at the same commit, same weights, prompts, seeds, sizes and frame counts, bf16, PyTorch SDPA, no FP8, no compile
(`reference/ltx_ref.py`, `results/rtx_pro_6000/results_gpu.csv`):

| job | 2 x p150a | RTX PRO 6000, stock / models kept on the GPU |
|---|---|---|
| 720p, prompt already encoded | 20.51 / 20.58 s | 15.72 / 15.75 s / 13.57 / 13.62 s |
| 720p, new prompt | ~33 s | 17.05 / 17.19 s / 14.20 / 14.25 s |
| 1080p, prompt already encoded | 55.03 / 54.84 s | 36.45 / 36.91 s / 34.16 / 34.20 s |
| I2V 1080p, 121 frames | 59.61 / 59.63 s | 38.65 / 38.17 s (total incl. prompt) |
| one pass 249 / 377 frames, 1080p | 178.9 / 316.3 s | 89.91 / 159.34 s |
| energy per 720p clip | cards ~1.2 Wh (from W x s) | GPU board 2.0-2.1 Wh |

The GPU host had 1 TB RAM (the stock pipeline rebuilds its models from RAM every call); GPU energy is board power only,
ours is the cards (whole-PC figures above). A tuned GPU path (FP8, FlashAttention) would be faster.

## Correctness gates

Reference = the official `ltx_pipelines` on the host CPU (`check/cpuref.py`, `check/cpuref_i2v.py`; bf16 as shipped,
checked against fp32 on selected blocks). Teacher-forced: every reference denoiser call's exact inputs run once on the
cards (`check/pcc_dit.py`, `results/pcc.jsonl`, `results/pcc_gemma.jsonl`).

| component | PCC |
|---|---|
| Gemma 4 text encoder, bf8: 49 hidden states / final video and audio embeddings | >= 0.9975 / 0.99996, 0.99993 |
| transformer stage 1 (8 steps), video / audio | 0.990 0.988 0.978 0.984 0.967 0.979 0.996 0.999 / >= 0.9993 |
| transformer stage 2 (3 steps), video / audio | 0.987 0.996 0.999 / 0.9993 0.9999 0.99998 |
| spatial upsampler | 0.986 (fp32 CPU arbiter: reference 0.9989, cards 0.987) |
| conv video VAE decode / audio decode + vocoder | 0.99996 / 0.9964 |
| I2V: image latent, transformer stage 1 / 2 | 0.99964 / >= 0.99828 / >= 0.99898 |
| I2V free-running clip vs the CPU clip, same seed | 0.978 (25.1 dB): the same clip, motion details drift |

**Text-to-video does not reproduce the reference clip from the same seed.** Per step the port matches (above), but the
first steps at sigma ~1 differ by 1-3 % and the 11-step ancestral sampler then lands on a different, equally valid
composition (final frames 14.2 dB vs the reference). Ruled out: weight dtype, matmul/attention fidelity, RoPE,
positions, the keyframe marker, the sampler and its noise streams (bit-identical start noise). A few outlier channels of
the bf16 residual stream differ more than bf16 noise in every block; cause not found. With a start image the trajectory
is anchored and the clip matches (25.1 dB). Same-config repeats are bit-identical (sha256 of the mp4).

## What changed against tt_dit's LTX-2.3 port (patches)

- LTX-2.5 checkpoint: no FFN biases (`ff_bias: false`), the keyframe marker added to the first frame's tokens like the
  reference, the **ancestral (SDE) Euler sampler** with the reference's own noise streams (tt_dit had 2.3's
  deterministic step), bf8 weights (`LTX_DIT_WEIGHTS=bf8`; attention `to_out` stays bf16 for the fused kernel), the aux
  checkpoint and the local upsampler.
- **Gemma 4 12B on the cards** (new `model_gemma4.py`): sliding and global layers, global layers with one KV head and
  k = v, partial proportional RoPE, unscaled v norm, per-layer scalar.
- **Key-padding masks on a 1-row mesh**: attention attended tile-padding tokens (video 364 -> 384, audio 26 -> 32):
  stage-1 PCC 0.940 -> 0.990. Upsampler: zero padding + masks instead of replicate padding for odd heights (0.974 ->
  0.986).
- 1080p: the VAE decodes in overlapping width tiles (a full-width decode wants one 6 GB buffer).
- Borrowed conv3d blockings (`LTX_CONV3D_BORROW=1`), clamped to the layer's frame count; off below 121 frames.
- Image-to-video like `ltx_pipelines` (frame-0 conditioning, pinned tokens, per-token timesteps), overlap chaining,
  and the 1080p memory work for long single passes: one shared mask in query chunks, CCL buffers freed per stage
  (`LTX_CCL_FREE`, +16 s per clip, long passes only), two-row I2V AdaLN and early frees (all bit-identical at 121
  frames).
- Development switches, off unless set: `LTX_SDPA_HIFI4`, `LTX_MM_HIFI4`, `LTX_SDPA_CHUNKS`, `LTX_DUMP_*`,
  `LTX_AUX_EVICT`, `LTX_FFN_CHUNK` (not bit-identical), and a frozen-audio mode for a lip-sync test.

## Tried and not kept

- Attention chunks: 512x256 22.2 / 22.6 s, 128x512 23.7 / 23.7 s (720p; stock 256x256 22.0), 512x512 does not build.
- HiFi4 for attention and matmuls: no PCC change.
- Tracing the text encoder: the second clip's replay hung the cards (reset); encoder runs untraced.
- Gemma 4 resident next to the 720p transformer: out of memory at stage 2 (hence the swap above).
- The diffusion video decoder (DiffVAE): same latents through both decoders on the CPU are 33-37 dB apart; slightly
  crisper in 100 % crops, no visible win at normal size. Porting it needs a neighbourhood-attention op ttnn does not have
  (estimate 2-3 days, not done). The conv decoder is used.
- Frozen-audio lip sync with the distilled model: the mouth follows vowels and some m/b closures; many syllables miss.

## Known issues (said plainly)

- **Three card hangs, each fixed by a reset + smoke test:** the borrowed conv3d table at a 25-frame shape (twice; the
  table is now off below 121 frames) and a traced text encoder (now untraced). Use a timeout around every card job.
- 377-frame single passes run only as the first clip of a process (fragmentation); 241 frames would need the 6x slower
  masked attention (lengths without any mask: 121, 249, 377, 505 frames).
- Heights must be multiples of 64: 448 / 704 / 1088. We crop 1088 to 1080 for 1080p output.
- The model's audio is generated, not conditioned on the image; in chained clips it is cross-faded.

## Layout

| path | what |
|---|---|
| `run.sh` | setup / aux / smoke / gen / py; every mount and flag is in this script |
| `patches/` | 12 commits on top of `../wan22-a14b-p150x2/patches` (`git am --keep-cr`) |
| `harness/gen_ltx.py` | the runner: job list in, mp4 + `runs.jsonl` lines out (I2V, segments, overlap, single pass) |
| `harness/prep_img.py`, `mkaux.py` | start-image preprocessing (official code, CPU venv); the aux VAE/audio checkpoint |
| `harness/jobs3.json`, `prompts8.json`, `jobs_i2v_*.json`, `i2v_moat_prompt.txt` | the counted prompts; the 8 test prompts; the counted I2V jobs (our start image is not included) |
| `harness/ltx_powersum.py`, `check/powerlog2.sh` | energy per clip from the power log |
| `check/cpuref*.py`, `refdecode.py`, `pcc_*.py`, `cmp_i2v.py` | CPU references and the PCC tools |
| `check/cutdetect.py`, `seam.py`, `flowcheck.py` | cut detector, seam and optical-flow checks for long clips |
| `check/sdpa_*.py`, `vae_prof.py`, `venv.sh`, `smoke.py` | attention micro-benchmarks, VAE profile, the CPU venv, card smoke test |
| `reference/` | the RTX PRO 6000 harness (`ltx_ref.py`, job files, pod entrypoint, `analyse.py`) |
| `results/` | `runs.jsonl` (every run), `pcc.jsonl`, `pcc_gemma.jsonl`, `power_summary.jsonl`, `fresh720.json`, `rtx_pro_6000/results_gpu.csv` |
| `LICENSE-LTX-2.x` | the LTX-2.x Community License (full text) |

The stage tables above come from the port's work log; `results/runs.jsonl` holds the per-clip lines they were taken from
(stage times, `LTX_*` flags, run names). Only server paths inside the result files were rewritten to this package's
container layout; no value was changed.

## Licences

- This package's code: Apache-2.0 (`LICENSE` at the top of this repository). `patches/` modify Tenstorrent tt-metal
  files (Apache-2.0).
- **LTX-2.5 weights and architecture: LTX-2.x Community License** (Lightricks; full text in `LICENSE-LTX-2.x`). The
  patches implement LTX-2.x's architecture in tt_dit and are distributed under that licence's terms as well: the full
  licence is included, its Attachment A (use restrictions) applies to anyone using this code with LTX-2.x, and every
  changed file is listed in the patch headers ("changed files" notices). Free for entities under USD 10 M annual
  revenue (section 2.1). Outputs must be disclosed as machine-generated (Attachment A 5). Do not remove watermarking,
  provenance or latent-disclosure features (section 6); this port removes none.
- Gemma 4 text encoder: its own terms (Google's Gemma terms), shipped by Lightricks inside the LTX-2.5 repository.
- No weights are part of this package.
