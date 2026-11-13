# Exit the Moat

Code, benchmarks and results behind the YouTube channel
[Exit the Moat](https://www.youtube.com/@ExitTheMoat): can you climb out of the CUDA moat, what does it cost, and is
the other side actually better?

Every number in a video comes from the code and the `results/` data in this repo.

- YouTube: [@ExitTheMoat](https://www.youtube.com/@ExitTheMoat)
- Hugging Face: [exit-the-moat](https://huggingface.co/exit-the-moat) (converted weights)
- Patreon: [Exit the Moat](https://patreon.com/Exitthemoat) (support the channel)

## Episode 3 in numbers: Qwen-Image-2512 on 1 and 2 Tenstorrent p150a

Text-to-image with the full base model (no turbo version), 8-bit (BF8) weights, 50 steps, guidance 4, full HD (made at
1920x1088, cropped to 1920x1080), text encoder, transformer and VAE all on the cards, 2026-11-12. Pictures per hour =
a counted queue of 20 pictures (5 prompts x 4 seeds) after a warm-up, two separate starts per setup. Details, method,
patches and the commands: [`qwen-image-2512-p150x2/`](qwen-image-2512-p150x2) (`results/table.py` prints these numbers
from its result files).

| setup | pictures/h, start 1 / start 2 | s per picture |
|---|---|---|
| one model split over both cards (tensor split) | **53.68 / 53.54** | **67.06** |
| one whole model per card, two pictures at a time | **55.84 / 55.77** | 127.8 |
| each card computes one guidance branch | 38.00 / 38.00 | 94.74 |
| one card alone | 29.07 / 29.09 | 123.82 |
| tensor split, 1024x1024 | 105.45 / 105.37 | 34.08 |
| two copies, 1024x1024 | 109.93 / 110.01 | 64.37 |
| energy per full-HD picture (tensor split) | 4.35 Wh on the two cards, 11.57 Wh at the PSU (whole box) | |
| against stock diffusers on an RTX 4080 SUPER, same seed and start noise (5 pictures) | PSNR 19.09 dB, SSIM 0.7225, CLIP 0.9338 | |

Same layout and subject as the RTX pictures; 50 denoising steps amplify tiny numerical differences, so 25-29 dB is
already the noise floor of "the same picture" (see the package README). At the start of the work (config v2) the
tensor split made 34.88-39.44 pictures an hour (four starts).

For comparison, a forum post reports 61.0 s per picture for Qwen-Image-2512 at 1024x1024 and 50 steps on a GB10 box
(published, not our run:
[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/image-diffusion-speeds/369095)).

## Episode 2 follow-up (12 Nov 2026): 4 parallel chats on 2 x p150a

After the video, the 2-card setup of Episode 2 got batching for several chats at once and a new GDN verify kernel
(WY / block form). Same model, same lossless DFlash2 output, same `serve.sh --cards 2`; the Episode 2 release stays one
flag away (`--ep2`). Measured through `serve.sh` with `check/steady.py` (512 tokens out, median of 3 rounds after a
warm-up), 12 Nov 2026. Details, the switches and the development runs:
[`qwen38-27b-p150-dflash2/`](qwen38-27b-p150-dflash2) (section "Update (12 Nov 2026)").

| total tok/s while all chats decode | 1 chat | 2 chats | 3 chats | 4 chats |
|---|---|---|---|---|
| `--ep2` (the Episode 2 release) | 156.4 | 174.1 | 226.3 | 264.5 |
| `--no-wy` (Episode 2 + batching) | 156.0 | 249.5 | 327.1 | 382.3 |
| **default `--cards 2`** (+ WY verify kernel) | **157.5** | **274.7** | **363.0** | **459.1** |
| default, including the first-token wait and queued prefills | 145.9 | 237.7 | 302.1 | 363.6 |

So 4 chats at once went from 1.7x to 2.9x the speed of one chat (about 115 tok/s each). Checks of the default start:
outputs exact 5/5 against the reference and 5/5 between two separate starts, GSM8K 59/60 (4 workers), needle in a
haystack 6/6. The `steady.py` harness differs from the Episode 2 table below (other prompts and output length), so
compare rows within one table.

## Episode 2 in numbers: Qwen 3.8 27B with DFlash2 on 1 and 2 Tenstorrent p150a

Lossless speculative decoding, one switch per setup, 2026-11-09. Decode = mean over the same 8 coding prompts (about 580 tokens each), one request at a
time unless the row says otherwise, gated on two starts and rebuilt from the package's README. Details, method and checks:
[`qwen38-27b-p150-dflash2/`](qwen38-27b-p150-dflash2).

| | 1 x p150a | 2 x p150a |
|---|---|---|
| decode, one developer | **75.8 tok/s** | **140 tok/s** |
| 4 developers at once, each | - (one request at a time) | **56.2 tok/s** (49-62) |
| 4 developers at once, total | - | **225 tok/s** while decoding, 191 tok/s including the wait for the first token |
| time to first token, 4 developers at once | - | 0.7 s median, 1.2 s for the last one |
| 8 developers at once, each | - | **26.4 tok/s** (24-32) |
| 8 developers at once, total | - | **211 tok/s** while decoding, 185 tok/s including the wait for the first token |
| time to first token, 8 developers at once | - | 1.3 s median, 2.6 s for the last one |
| time to first token, 1k prompt | 0.39 s | 0.27 s |
| time to first token, 8k prompt | 3.1 s | 1.8 s |
| time to first token, 32k prompt | 14.5 s | 8.3 s |
| outputs exact against the reference | 5/5 | 5/5 |
| GSM8K (60) | 60 / 60 | 59 / 60 |
| needle in a haystack | 6 / 6 | 6 / 6 |

The 33 tok/s per developer quoted in the video comes from the separate plain-decoding vLLM build of Episode 2, which is
not part of this package.

The 80 tok/s for one card in the video used a different set of code prompts: on that set this package's 1-card server
also measures 79.6-80.2 tok/s, against the 75.8 tok/s above on the 8 prompts in `check/`.

For comparison, my own RTX PRO 6000 run (NVFP4 + DFlash2, coding prompts, a different harness) does 239 tok/s for one
developer.

## Episode 1 in numbers: Qwen 3.8 27B on 1 and 2 Tenstorrent p150a

Plain decoding with vLLM-TT, 2026-11-01. Details, method and patches: [`qwen38-27b-p150x2/`](qwen38-27b-p150x2).

| | 1 x p150a | 2 x p150a |
|---|---|---|
| 1 developer, 1k prompt | 7.4 tok/s | 14.6 tok/s |
| 8 developers, 4k prompts | - (batch 1 only) | 11.2 tok/s each, 58 total |
| GSM8K (60) | 59 / 60 | 59 / 60 |

## Episode 0 in numbers: Fish Audio Drama 3 (36B voice model) on one Tenstorrent p150a

One card, PCIe Gen3 x16, 8 test lines, 2026-10-31. RTF = compute seconds per second of audio (lower is better).
Details, method and run files: [`fish-drama3-p150/`](fish-drama3-p150).

| | p150a, this port | original model, rented RTX PRO 6000 |
|---|---|---|
| real-time factor, one voice | **0.381** (0.404 over the whole 47-line narration) | 0.173 |
| time to first audio | **0.41-0.47 s** | 0.16-0.20 s |
| many voices at once (audio s per wall s) | 9.51 (16 slots, narration chunks) | 17.7 (8 concurrent requests) |
| board power while speaking, one voice | mean 196 W | mean 263-270 W |
| price in Germany, incl. 19 % VAT | €1,613 (paid €1,355.67 + import VAT) | ~€21,000 (idealo.de, 31 Oct 2026; under €15,000 on 1 Sep 2026) |

Over the port: 14.7 RTF at the first end-to-end run, 0.853 at the end of "day 2", 0.381 after four more days of
optimization.

## Episodes

| # | Topic | Folder |
|---|---|---|
| 3 | Episode 3 · Qwen-Image-2512 on 2 Tenstorrent Blackholes: THE RENDER SPRINT (full-HD pictures on 1 and 2 p150a, four ways to split the work, the music-video prompts) | [`qwen-image-2512-p150x2/`](qwen-image-2512-p150x2) · video: [YouTube](https://youtu.be/pllVxuibc_g) |
| 2 | Qwen 3.8 27B on 2 Tenstorrent p150: optimization results (DFlash2 on 1 and 2 cards, one switch per setup; follow-up 12 Nov: 4 parallel chats, 459 tok/s total) | [`qwen38-27b-p150-dflash2/`](qwen38-27b-p150-dflash2) · video: [YouTube](https://youtu.be/TFR_2J9H__Q) |
| 1 | Qwen 3.8 27B on 2 Tenstorrent p150: getting started (vLLM-TT on 1 and 2 cards, the team benchmark, the patches) | [`qwen38-27b-p150x2/`](qwen38-27b-p150x2) · video: [YouTube](https://youtu.be/3AvhL99p77M) |
| 0 | I almost returned two Tenstorrent AI cards. It was the cable. (Fish Audio Drama 3 on one p150) | [`fish-drama3-p150/`](fish-drama3-p150) · weights: [Hugging Face](https://huggingface.co/exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150) · video: [YouTube](https://youtu.be/mpleglhA_Dg) |

New episodes are added here when they are published.

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights follow their own licences (Drama 3: Fish Audio Community License,
see [`fish-drama3-p150/`](fish-drama3-p150)). No Qwen-Image weights are in this repo; the 8-bit conversion happens in
your own cache.
