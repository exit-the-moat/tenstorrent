# Review: ep1 tt-metal patches (qwen36 on 2 x p150a, TP=2), 2026-11-01

Read-only review. Base compared against: the image's own source
(`vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af`, read from its overlay layer;
`orig/gdn_tp.py` is byte-identical to it). Files: `/opt/ep1/patches/gdn_tp_v5.py`, `bf4/*`, `patch_*.py`, `*.diff`,
`/opt/ep1/serve/start-ep1.sh`, the start logs.

## Default-off check

With no env set, every patched line evaluates to the stock value (`_fuse_out_mmrs_prefill` unchanged,
`_gdn_conv1d=True`, `_gdn_mc=L1_MEMORY_CONFIG`, no typecast, no extra kwarg; `_W8=bfloat8_b`). Same ttnn op
sequence, same tensors. The only difference is an `os.environ` lookup per `forward_prefill` call. Confirmed: no
behaviour change when unset.

All three GDN patches touch **prefill only**. Decode (the 14.6 tok/s) runs the same code as stock; only host sampling
(a serve setting) changes decode.

## Findings (ranked)

1. **Medium, numerics: `QWEN36_GDN_UNFUSED_RS` is not "fp32 like the fused op"** (`gdn_tp_v5.py:636-650`,
   `_row_proj` 374-395). `_row_proj` calls `ttnn.linear` with no `dtype`, so each device's partial comes out
   **bf16** (fp32 dest accumulation, then packed to bf16). Only after that is it typecast to fp32. The fused
   `matmul_reduce_scatter_prefill(..., ttnn.float32)` writes fp32 partials. So the patch gives "bf16-rounded
   partials, summed in fp32", not the fused op's numerics. Also, the `dtype=ttnn.float32` kwarg passed to
   `tt_all_reduce` does nothing on a 1xN mesh: `ccl.py:169-191` takes the reduce_scatter branch, which never reads
   `dtype` (only the TG all-reduce branch does). At 2 devices the error is small (same order as the stock 1-card
   path, which outputs bf16), and GSM8K 59/60 agrees. The stock comment says bf16 here hurt PCC at 4 devices (0.69),
   so it isn't safe to assume for 1x4/1x8. Fix: pass `dtype=ttnn.float32` to the prefill `ttnn.linear` in
   `_row_proj` when the override is active (the comment there already plans for an fp32 [seq,dim] DRAM output),
   drop the typecast and the dead kwarg, then check PCC against the fused path on 1x4 and against a torch
   reference on 1x2 at 2k/8k/32k.
2. **Low-medium, wrong scope: the flag also acts where the fused path was never active.** `_f32` is read from the
   env alone. On a 1-card mesh, `tt_all_reduce` returns its input for `[1,1]`, so the GDN prefill output turns
   **fp32** instead of stock bf16 (extra typecast plus 2x DRAM, and a dtype change downstream). The same happens
   with `gdn_out_weight_memcfg` set (out-sharded). Fix: in `__init__`, set
   `self._unfused_rs = env and not self._out_sharded and args.num_devices > 1`, and use it in both places.
3. **Low, memory hygiene:** after `partial = ttnn.typecast(partial, ...)`, the bf16 tensor is freed only through
   Python refcount. The file otherwise deallocates explicitly. Add `ttnn.deallocate`. Peak per 2048 chunk: about
   21 MB bf16 + 42 MB fp32 in DRAM, not a problem. 32k prompts go through the 2048-chunk path
   (`prefill_chunked_peruser`), so per-chunk memory doesn't grow with prompt length.
4. **Low, perf:** `PREFILL_DRAM` moves qkvzab, its slices and the FIR conv output from L1 to DRAM, which costs
   DRAM bandwidth in prefill. `UNFUSED_RS` gives up the matmul/RS overlap; the stock comment puts it at about -2 s
   TTFT at 128k (measured on 1x4). Expect slower TTFT, not slower decode. Link count is fine: two BH devices are
   named "P300" in `tt_ccl`, which gives 2 links, the same as the fused op.
5. **Low, flag handling:** `CONV1D` and `UNFUSED_RS` are read in `__init__`, but `PREFILL_DRAM` and `_f32` are read
   on every call (lines 496, 639). Read them all once in `__init__`. `QWEN36_GDN_CONV1D` set to `""` or `true`
   disables conv1d (anything except `"1"` does).
6. **Low, BF4 (`patch_bf4.py`):** the docstring says "cache file names don't carry the dtype". That's wrong for
   `ttnn.as_tensor` (`ttnn/operations/core.py:716` appends `_dtype_<X>_layout_<Y>`), which `tpc.shard_w` and the
   lm_head use. A separate cache is harmless but not needed. The inserted header (`_W8 = ...` between imports) trips
   ruff E402 on the imports after it; mlp.py gets 3 blank lines. 60/60 on 60 questions can't show quality parity:
   the 95% interval is several points wide.
7. **Hygiene: superseded files.** `qwen36_vllm.py` (QWEN36_PREFILL_CHUNK), `ep1-conv-split.diff`/`gdn_tp.py`,
   `gdn_tp_fir*.py` and `ep1-prefill-chunk.diff` were not used in start 6. `ep1-gdn-v5.diff` is a normal-format
   diff with no paths, so `git apply` can't use it. Regenerate it as a unified git diff against tt-metal `de59f8a`
   with `a/models/demos/blackhole/qwen36/tt/gdn/tp.py` paths.
8. **Publishability:** the patch files contain no tokens, IPs, hostnames or user names. `start-ep1.sh` has host
   paths (`/opt/ep1`, `/opt/llm/hf`, `/opt/tenstorrent/start-tt-qwen38.sh`), the `/dev/sda` write throttle and
   port 8085: parametrise or scrub them. The modified files keep Tenstorrent's SPDX header. Under Apache-2.0
   §4(b), add a "modified by" note to each one. `patch_*.py` have no SPDX header: add
   `SPDX-License-Identifier: Apache-2.0` with your own copyright line.

## Verdicts

| Patch | Verdict |
|---|---|
| `QWEN36_GDN_CONV1D=0` (FIR) | Publish as-is. It reuses the stock FIR path (already used for masked buckets), and the numerics are equivalent. |
| `QWEN36_GDN_PREFILL_DRAM=1` | Publish as-is. It only changes memory placement; the numerics are identical. |
| `QWEN36_GDN_UNFUSED_RS=1` | Fix first (findings 1-3), or publish the measured version with its comment corrected to "fp32 sum of bf16 partials". After a fix, re-run GSM8K. |
| Host sampling | Serve setting, nothing to review. |
| `QWEN36_NAIVE_BF4=1` | Publish as an experiment once finding 6 is fixed. Don't upstream it. |
| conv-split, prefill-chunk, fir, fir_dram files | Don't publish (or archive them as history). |

## Upstream

**Open an issue with a minimal repro, not a PR.** An env-flag PR would likely be rejected. A real fix would
auto-select: check that the RS worker block fits in `compute_with_storage_grid_size()` and fall back to unfused
if it doesn't; base L1 vs DRAM on per-device tensor size at TP=2 (start 4 asked for 20 MB in L1 across 110 banks);
use an fp32 matmul output; add a 1x2 test; let `qwen36_vllm.py` accept (1,2) for on-device sampling.

Draft:

> **qwen36: TP=2 on 2 x p150 (P150x2, mesh (1,2), FABRIC_1D) fails in prefill warm-up**
> Stock 0.20.0-de59f8a fails in sequence: (1) `ttnn.conv1d` GDN prefill: static CBs clash with L1 buffers on
> [0-0 - 10-5]; (2) FIR conv in L1: OOM (20 MB across 110 banks); (3) `matmul_reduce_scatter_prefill` grid (8,8),
> RS offset (0,8): "No core coordinate found at (0, 10)". Also, on-device sampling rejects (1,2). Workarounds
> (FIR conv, DRAM prefill projections, unfused out-proj + RS) run it: GSM8K 59/60, 14.6 tok/s at batch 1.
> Patch and logs attached. Would you accept grid/L1-aware auto-selection?
