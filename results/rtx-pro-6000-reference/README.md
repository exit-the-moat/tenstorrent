# Drama 3 reference on one rented RTX PRO 6000 (2026-10-30)

The original `fishaudio/drama-3` checkpoint (BF16 + FP8 experts, HF revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`)
served with the official SGLang-Omni server (`sgl-omni serve`, all defaults) on one rented GPU. Same 8 test lines
(`text` = spoken form from `fish-drama3-p150/eval/lines.yaml`) and the same voice reference (`ref_v2.wav` + `ref_v2.txt`)
as the Tenstorrent port. All numbers below come from the files in `results/` (`summary.json` is written by `summarize.py`).

## Measured values

| | pass 1 (`seq`) | pass 2 (`seq2`) |
|---|---|---|
| lines / audio s | 8 / 98.1 | 8 / 97.2 |
| real-time factor (sum gen s / sum audio s) | **0.173** | 0.173 |
| RTF per line | 0.171-0.178 | 0.171-0.178 |
| first audio, first measured line | 0.165 s | 0.157 s |
| first audio, later lines | 0.156-0.201 s | 0.152-0.162 s |
| frames (server usage count) | 2,112 | 2,093 |
| ms per frame, whole request (gen s / frames) | 8.04 | 8.02 |
| ms per frame after first audio (median, streaming arrival rate) | 7.70 | 7.72 |
| GPU memory used while speaking (nvidia-smi) | 76,769 MiB | 76,769 MiB |
| GPU board power while speaking (0.5 s samples) | mean 262.9 W, min 213.6, max 269.4 (n 34) | mean 269.8, min 259.3, max 273.9 (n 34) |
| Whisper match (mean, min) | **0.927 (0.837)** | 0.932 (0.837) |

- **Very first request after server start** (warm-up line, not counted): first audio 3.78 s, RTF 0.986. That request
  also encoded the reference voice (reference-encoder cache miss; all later requests hit the cache) and paid the
  first-request warm-up.
- **Model load**: download from HF 27.5 s (41.5 GB, both files; RunPod DC network). Server start to `/health` 200:
  **88.8 s**, of which the weight load itself was 19.13 s (`Load weight end. elapsed=19.13 s ... mem usage=36.07 GB`),
  prefill CUDA graph capture 31.7 s, decode graph capture 4.1 s. The weights had just been downloaded, so they were most
  likely read from the page cache, not from disk (the container can't drop caches).
- **Idle after load**: 74,975 MiB used, 98.2 W (`smi_idle.csv`, one sample).
- **Memory is a reservation, not a need**: the engine takes 0.70 of the GPU by default (KV cache for 663,262 tokens,
  29.1 GB), the vocoder 9.5 GiB share, preprocessing 6 GiB cap. Weights are 36.07 GB. Per process (idle):
  engine 69,750 MiB, preprocessing 2,712 MiB, vocoder 2,494 MiB (`smi_apps_idle.csv`).
- **Codec**: on the same GPU, in its own process (default placement), fp16 decode, window CUDA graphs.
- **Bonus, 8 lines at once** (`c8`, 8 concurrent streaming requests): 96.2 s of audio in 5.42 s wall =
  **17.7 audio s per wall s**; RTF per stream 0.378-0.504, first audio per stream 1.37-1.44 s; power mean 275.4 W
  (min 135.5, max 329.1, n 11); Whisper 0.925 (0.818).

## Method

- Pod: RunPod **SECURE** cloud (community had no RTX PRO 6000 stock), `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  (96 GB, 600 W limit, sm_120), driver 595.91.07, US location. Image `hongccc/sglang-omni:dev` (the image the
  installation doc names; Docker Hub digest `sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985`,
  pushed 2026-10-20), with SGLang-Omni checked out at **`1b1f897df32c0a9f5d7e5d9d66d02e5ec00d83cd`** and installed
  `pip install --no-deps -e` (as the image's own entrypoint does). torch 2.14.1+cu130, CUDA 13.0.3, sglang 0.5.22,
  flashinfer 0.7.1, triton 3.8.0, transformers 5.19.0, Python 3.12.15 (`results/versions.txt`).
- Server: `sgl-omni serve --model-path fishaudio/drama-3 --host 127.0.0.1 --port 30000`, nothing else changed. Its log
  says attention backend triton, `Drama 3 codebook sampling: PyTorch, untested GPU sm_120` and `tower fused layers: off,
  untested GPU sm_120` (the Triton sampler and fused tower are Hopper-only; this is the official default path on this GPU).
- Client (`pod/bench.py`, on the pod, localhost): `POST /v1/audio/speech`, `stream: true`, `stream_format: sse`,
  `response_format: pcm`, the reference sent inline as base64 WAV with its transcript, `max_new_tokens: 600` (the card's
  max_frames). Sampling = server defaults: temperature 1.0, top_p 0.9, no top-k cap on the main token, codebook heads
  top-k 30, the repetition-aware redraw. These match the defaults of our port (`temperature=1.0, top_p=0.9, top_k=None`).
  Order: one warm-up line, then the 8 lines one after the other (pass 1), the same again (pass 2), then all 8 at once.
- Timing is client side: t0 just before the POST; first audio = first non-empty PCM delta; gen = the `done` event.
  RTF = gen s / audio s. Frames = the server's `usage.output_tokens`; it matches samples / 2048 exactly.
- Power and memory: `nvidia-smi --query-gpu=power.draw,memory.used,... -lms 500` with the epoch prepended to each line
  (`results/smi.csv`), averaged over the samples inside each line's [start, end] window.
- Whisper: `score_refgpu.py` on the AI server, the same faster-whisper call as the card (`host/transcribe.check`) and
  the same `match_written` ratio as `day7_scoreboard.py`. It ran in the ep0 TT image **without** `/dev/tenstorrent`
  mounted, so the card was not touched. WAVs and `sanity.json` are also in `/opt/ep0/refgpu/` on the server.

## Differences from the card's method (state them on screen if the numbers are compared)

1. **No seed.** The official server refuses a `seed` (`speech_request.py`: `UNSUPPORTED_PARAMS = ("seed", ...)`), so
   the takes are not seed 0 and not reproducible; pass 1 vs pass 2 shows the spread (frames 2,112 vs 2,093, Whisper
   0.927 vs 0.932). The card's runs are seed 0.
2. **Different sampler implementation**: PyTorch sampler of SGLang-Omni on GPU vs our device sampler. Same rules and
   settings, not the same random draws.
3. **No backbone / decoder split.** The server doesn't expose per-stage time. "ms per frame" here is the whole
   pipeline (backbone + codebook tower + sampling + scheduling, with the vocoder running in parallel in its own
   process), measured from the client. The card's 12.6 + 8.9 ms are device-side backbone and tower times. Not the same
   quantity.
4. **TTFA includes HTTP + SSE on localhost** and the server's streaming window; the card's ttfa_s is measured in-process.
   The reference is encoded once (warm-up) and cached, like the card's prefix cache.
5. **"First line" differs**: here the very first request after start (3.78 s) was a warm-up line; the counted first line
   ran on a warm server. On the card "TTFA first" is the first of the 8 lines.
6. **Power is the whole board** (nvidia-smi `power.draw`), as for the card's board W; there is no chip-only figure.
7. **Model load**: weights most likely came from the page cache (just downloaded), the card loads from its SSD.
8. Checkpoint precision: the original FP8 experts + BF16 rest, vs BFP4/BFP8 on the card.

## Cost

| | |
|---|---|
| pod id | (removed; deleted with `DELETE /v1/pods/...`, HTTP 204; pod list empty afterwards) |
| price | $2.49/h (secure) |
| lifetime | 17:23:34 to 17:26:46 UTC = 3 min 12 s (the image was cached on the host, so it booted in seconds) |
| rental cost | **$0.129 total** |

The reference voice was sent to the pod only over SSH (scp), never through the HTTP proxy, which served only
`status.txt` and `ssh.log`. The voice file and all generated WAVs were deleted on the pod before it was deleted.

## Files

- `pod/run.sh` (pod entrypoint), `pod/bench.py` (client), `summarize.py` (summary table), `score_refgpu.py` (Whisper).
- `results/`: `seq/`, `seq2/`, `c8/` (8 WAVs at 44.1 kHz + `timing.json` + `sanity.json` each), `warmup.json/.wav`,
  `load.json`, `smi.csv`, `smi_*.csv`, `server.log`, `bench.log`, `versions.txt`, `download.log`, `status.txt`,
  `score.log`, `summary.json`. The WAVs are cloned from Sebastian's voice: private, don't publish.
