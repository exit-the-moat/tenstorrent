# Fish Audio Drama 3 on one Tenstorrent p150 (Episode 0)

**Built with Fish Audio.** A port of [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) to a single
Tenstorrent Blackhole p150a with tt-metal, written for Episode 0 of [Exit the Moat](https://www.youtube.com/@ExitTheMoat),
where this port narrates the video. Not an official Fish Audio or Tenstorrent release; neither company endorses it.

- Video: [I almost returned two Tenstorrent AI cards. It was the cable.](https://youtu.be/mpleglhA_Dg) on [Exit the Moat](https://www.youtube.com/@ExitTheMoat)
- Converted weights (tt-metal weight cache, BFP8 / BFP4): [exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150](https://huggingface.co/exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150) on Hugging Face
- Support the channel: [Patreon](https://patreon.com/Exitthemoat)
- Measured results: [`results/drama3-runs/`](../results/drama3-runs), [`results/drama3.csv`](../results/drama3.csv) and the
  GPU reference [`results/rtx-pro-6000-reference/`](../results/rtx-pro-6000-reference)

## What runs where

The 40-layer MoE backbone (a Qwen3.5-MoE-style model, run on tt-metal's qwen36 model code: gated attention + gated DeltaNet linear attention, 256 experts, top-8 plus a
shared expert), the codebook tower, the sampler and the audio codec decoder all run on the card. The host does
tokenisation, the prompt and the streaming / playback bookkeeping.

## Results

All on one p150a (PCIe Gen3 x16, card 0 of two, the other card idle), the 8 test lines in `eval/lines.yaml` unless
noted, 2026-10-31. RTF = compute seconds per second of audio (lower is better; below 1 is faster than real time).
Every number links to its run files.

### One voice: `run_gen.sh --interactive` (the low-latency preset)

| | value | run files |
|---|---|---|
| real-time factor | **0.381** (per line 0.373-0.412) | `port10/i_fin`, repeats `k_on_1`, `k_on_2` |
| time to first audio | **0.409-0.472 s** on every line | same |
| one frame (46.4 ms of audio), whole loop | 17.5 ms | same |
| playback from the first sound | gapless (0 underruns) | same |
| card DRAM | 31.29 GB of 33.9 | `port10/i_finpw` (tt-smi, 17 samples) |
| chip / board power while speaking | 43-94 W (mean 56.1) / 162-219 W (mean 195.9) | `port10/i_finpw/power.txt` |
| Whisper match vs the script, mean (min) | 0.908 (0.744; the min is the "120 Tensix cores" line Whisper mishears) | `port10/k_on_1.wcheck.log` |

The whole Episode 0 narration (47 lines, 399 s of speech) was spoken with this preset: RTF **0.404**, first audio
0.43 s median, 0 underruns, Whisper mean 0.975 (`narration-v20/`; the WAVs themselves are not published, the voice
is a clone of the author's).

### How it got there

| stage | RTF (8 test lines) | first audio | run files |
|---|---|---|---|
| first end-to-end run (backbone on the card, codec on the CPU) | 14.743 | 33.7 s | HANDOFF.md, Day 3 |
| traced backbone + tower, sampling and codec on the card, cached voice prompt (`--fast`, the end of "day 2" in the video) | 0.853 | 1.19-1.23 s (later lines) | HANDOFF.md (Day 4b) |
| port6: fused kernels, `--interactive` | 0.53-0.57 | 0.45-0.49 s | `port6/i_qdt` |
| port10: + codec traced on the card, same-precision levers, shared expert in the expert kernel, bfp8 tower MLP, fused sampler, weight prefetch | **0.381** | 0.41-0.47 s | `port10/i_fin` |

The lever-by-lever log (before / after per change, its quality check, what was tried and dropped) is
[`drama3/HANDOFF.md`](drama3/HANDOFF.md), newest section first. Run directories named there as `/opt/ep0/<stage>/...`
are the published folders under `results/drama3-runs/<stage>/` (WAVs and tensors left out).

### Many voices at once: continuous batching (`host/cbatch.py`)

Full queue of 32 jobs at t = 0, greedy, audio seconds produced per wall-clock second (port10, all levers "C4", see
`HANDOFF.md`):

| workload | B = 8, port9 baseline | B = 8 | B = 12 | **B = 16** | B = 24 |
|---|---|---|---|---|---|
| short: the 8 test lines x 4 | 4.19 | 6.19 | 7.04 | **8.05** | 7.86 |
| long: narration chunks x 4 | 4.93 | 7.01 | | **9.51** | 9.00 |

B = 16 is the ceiling for this queue; B = 27 runs out of card DRAM. Run files: `port10/f_short_*`, `f_long_*`,
`fb_*` with their `.wcheck.log`.

### The original model on a rented GPU, for comparison

The original `fishaudio/drama-3` (BF16 + FP8) on its official server (SGLang-Omni, all defaults) on one rented
RTX PRO 6000 Blackwell, same 8 lines and voice ([`results/rtx-pro-6000-reference/`](../results/rtx-pro-6000-reference)):

| | p150a, this port | RTX PRO 6000 | ratio |
|---|---|---|---|
| real-time factor | 0.381 | 0.173 | 2.2x |
| ms per frame, whole loop | 17.5 | 8.0 | 2.2x |
| first audio | 0.41-0.47 s | 0.16-0.20 s | ~2.5x |
| 8 voices at once, audio s per wall s | 6.19 (B = 8; 8.05 at B = 16) | 17.7 (8 concurrent requests) | |
| memory bandwidth (vendor spec) | 512 GB/s | 1,792 GB/s | 3.5x |
| price in Germany, incl. 19 % VAT | €1,613 (paid €1,355.67 incl. cable share and shipping, + import VAT) | ~€21,000 (idealo.de, 31 Oct 2026; under €15,000 on 1 Sep 2026) | ~13x |

Different shapes for the batched row (the GPU took a burst of 8 lines, the card a 32-job queue), so compare it
loosely.

### Load times

The converted weights (~25 GB) load onto the card in about 11 s from the page cache; from the SATA SSD in the test
server a cold load took 183-186 s, 154 s of which is the disk (163 MB/s). Details: `HANDOFF.md`.

## Quality gates

Every speed change was kept only if it passed:
- **Golden check** (`tests/check_paged_backbone.py`): per-layer PCC of the card's backbone against an FP32 CPU
  reference of Drama 3 (`reference/cpu_layer_ref.py`) on a fixed voice prompt plus 24 teacher-forced frames.
- **Batch numerics** (`tests/check_batch_numerics.py`) for the parallel paths: B streams against B = 1.
- **Multi-seed Whisper gate** (`tests/gate_seeds.py`, `tests/gate.py`): two lines, five seeds, transcript vs text.
- `--fast` (the preset the episode was rendered with) stays bit-identical to the Day 4b render.

`eval/score.py` adds word error rate and speaker similarity per line to `results/drama3.csv`.

## Stack (pinned)

| Part | Version |
|---|---|
| Card | 1 × p150a, firmware bundle 19.15.0, PCIe Gen3 x16 (Ryzen 7 5700G host) |
| Driver | tt-kmd 2.11.0, tt-smi 6.6.0 |
| tt-metal | `70222e90` (2026-10-03) + [`patches/tt-metal-70222e90.patch`](patches/tt-metal-70222e90.patch) (17 files of the qwen36 MoE / GDN / attention code, all changes opt-in), SFPI 7.84.0 |
| Toolchain | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af`, with `LD_LIBRARY_PATH` unset, `TT_METAL_HOME` on the new tree, `ARCH_NAME=blackhole` |
| Codec sources | [SGLang-Omni](https://github.com/sgl-project/sglang-omni) `1b1f897d`, `descript-audio-codec` 1.0.0 |
| Model | `fishaudio/drama-3` |

## Running it

Tested from scratch on 2026-10-31: a fresh clone of this repo, a new container, the weights from Hugging Face, no
reference voice. Times below are from that run (Ryzen 7 5700G, SATA SSD).

**Work directory.** The scripts use `/opt/ep0` (the directory used for the video), in about 90 files. If that path is
free on your machine, use it (or a symlink to your disk). If it isn't, rewrite the paths in your clone before running
`setup.sh`; this is how the reproduction test did it:

```bash
grep -rl /opt/ep0 fish-drama3-p150 --include='*.sh' --include='*.py' | xargs sed -i 's#/opt/ep0#/opt/mydir#g'
```

**Setup.**

```bash
export MODEL_DIR=/path/to/fishaudio/drama-3        # hf download fishaudio/drama-3 (accept its licence first)
fish-drama3-p150/setup/setup.sh                    # tt-metal 70222e90 + patch + this port, build (~20 min), codec
                                                   # and Whisper side folders, config shim, smoke test
pip install -U "huggingface_hub[cli]"              # for the next line only
hf download exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150 --local-dir /opt/ep0/cache/drama3   # optional
```

- Without the downloaded cache, the first run converts the original weights itself (slow, ~25 GB written). The
  `.tensorbin` files are tied to this tt-metal version and end up in `/opt/ep0/cache/drama3/P150/tensor_cache_bfp8`.
- The container mounts relabel their folders for SELinux (`:z` / `:Z`), including `MODEL_DIR`.
- The `pip` step for the Whisper side folder prints red "dependency conflicts" against ttnn (numpy 2 etc.). That is
  harmless: the side folder is only on the path for the Whisper check. So is the tokenizer "incorrect regex pattern"
  warning on every run (the port tokenizes with `tokenizers` directly).
- Two cards? `TT_VISIBLE_DEVICES=1 <command>` picks the second one; `setup/run.sh` passes it into the container.

**Say something.** A lines file is a JSON list of `{id, spoken, written}` (`spoken` is what the model reads,
`written` the reference for the Whisper check). `--ref none` uses the model's own voice:

```bash
echo '[{"id":"hello","written":"Hi I am working on my Tenstorrent card with Fish Audio","spoken":"Hi I am working on my Tenstorrent card with Fish Audio"}]' > /opt/ep0/hello.json
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
$D/run_gen.sh /opt/ep0/out/gen.log --interactive --ref none --lines-file /opt/ep0/hello.json --lines all --out /opt/ep0/out
```

The **first run takes about 10 minutes** with long silent stretches (codec compile, backbone build, card codec,
warm-up and trace capture, about 2 minutes each); later runs reuse the compile caches. In the test the line itself
took 1.6 s: RTF 0.462, first audio 0.40 s (second card, 110 cores).

**Voice reference (optional):** a 10-30 s mono WAV in `/opt/ep0/voice/ref_long.wav` with its exact transcript in
`ref_long.txt` (or pass `--ref <wav>`). Use your own voice or one you have the rights to; no reference is
published here.

```bash
$D/run_gen.sh /tmp/gen.log --interactive --lines all --out /tmp/out    # the 8 test lines -> /tmp/out/<id>.wav + timing.json
$D/run_gen.sh /tmp/gen.log --fast --lines close --out /tmp/out         # the episode-render preset
fish-drama3-p150/bench/solo.sh /opt/ep0/bench                          # one voice: --interactive, --fast, golden check
fish-drama3-p150/bench/ladder.sh /opt/ep0/ladder                       # parallel voices: continuous + static batching
```

`timing.json` holds per line: `ttfa_s`, `rtf`, the playback check (`windows`, `underruns`, `worst_late_s`), and
the per-frame backbone and tower times. The golden check needs the FP32 reference for your voice prompt first
(`reference/cpu_layer_ref.py`, about 70 GB of weights streamed one layer at a time; see the commands section
"For an agent that knows nothing about this work" in `drama3/HANDOFF.md`).

## Layout

| Path | What |
|---|---|
| `drama3/tt/` | the backbone and tower on the card: paged + traced decode, fused decode ops, quantizer and codec decoder on the card |
| `drama3/kernels/` | hand-written Tensix kernels (fused GDN decode, MoE top-k / combine / SwiGLU, attention pre / post, batch variants) |
| `drama3/host/` | generators (`generate_fast.py`, `batch.py`, `cbatch.py`), prompt, tower reference, streaming codec |
| `drama3/reference/` | FP32 CPU reference of the backbone, one layer in RAM at a time |
| `drama3/tests/` | golden / numerics / gate checks, profiles, benches, tuning tables |
| `drama3/run_gen.sh`, `run_check.sh` | the presets (`--fast`, `--interactive`, `--batch`) and the test launcher |
| `patches/` | the tt-metal changes outside the port's folder |
| `setup/` | setup, build, container launcher, smoke test |
| `bench/` | the benchmark scripts behind the tables above |
| `eval/` | the 8 test lines and the WER / speaker-similarity scorer |

`drama3/host/prompt.py` and `drama3/host/tower.py` are ports of SGLang-Omni code (Apache-2.0).

## Licence

Our code: Apache-2.0 (see the repository `LICENSE`); the tt-metal patch modifies Apache-2.0 files of
tenstorrent/tt-metal and says so in its header. Drama 3 itself and the converted weights are under the Fish Audio
Community License v1.0: read it before use. Among other things it forbids imitating an identifiable person's voice
without their consent and offering a hosted text-to-speech service without a separate licence from Fish Audio.
