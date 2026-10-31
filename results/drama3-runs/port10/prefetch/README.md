# port10 prefetch prototype: DRAM-core tensor prefetcher for the tower matmuls (card 1, 2026-10-31)

Bench: `models/demos/blackhole/drama3/tests/bench_tower_prefetch.py` (new file, not integrated; copy here).
Random weights with the tower shapes/dtypes, one layer = eh_h 2048x2048 bf16, q|gate|k|v 2048x9216 bfp8,
o_proj 4096x2048 bfp8, w1 / w3 2048x6144 bf16, w2 6144x2048 bf16 (112.6 MB), HiFi4 fp32-acc (the tower ckc).
Stand-in small ops between the matmuls (rms_norm / add on a [1,32,2048] row): 86.5-89 us of device time per
layer alone (`c_small_ops`), sized to the measured tower gaps. Whole frame = one trace, timed over 20-50 replays
(host wall clock per replay, device-bound). Card 1 (11x10 grid, 8 DRAM banks, fw 19.15).

## Numbers (all measured)

| variant | 1 layer, us | 13 layers, ms/frame | GB/s (useful bytes) |
|---|---|---|---|
| (a) today: interleaved weights, decode_linear 1D-mcast configs, matmuls only | 318 | 4.06 | 361 |
| (a) + small ops | 375 | **4.78** | 307 |
| (a2) same 64 cores / config as (b), interleaved, no prefetcher, + small ops | 444 | 5.63 | 260 |
| (b) prefetcher, ring 64, GCB 512 KB/core, request captured in trace, matmuls only | 282 | 3.62 | 405 |
| (b) same + small ops | 291 | **3.79** | 388 |
| (b) request queued ahead, 128 KB pages, + small ops | 290 | 3.76 | 390 |
| (b) request ahead, 128 KB pages, GCB 1 MB/core, + small ops | - | **3.64** | 403 |

Files: L13_cap.json, L13_ahead_p128.json, L13_ahead_p128_g1024.json, r8_cap.json, r8_ahead*.json, silu.json.

- Small-op hiding (13 layers): (a) exposes 55 us/layer of the small ops; (b) exposes 12.9 us/layer
  (8.6 us with the 1 MB GCB) of the 86.5 us they take alone, i.e. about 85-90 % hidden.
- Per frame: 4.78 -> 3.79 ms (-0.99 ms) with the captured request, -1.14 ms with GCB 1 MB + request ahead.
- Correctness: (b) vs (a2) bit-identical for every matmul checked (layers 0 and 12 of 13, plus 1-layer runs);
  (b) vs (a) also bit-identical (max abs 0.0); PCC vs torch fp64 0.99997 (bfp8) / 0.999997 (bf16).
  Fused SiLU on w1 (--silu 1): works, bit-identical to (a2), PCC 0.999994 vs (a) (default linear SiLU path).

What did NOT work / matters:
- Ring 32 (8 banks x 4 receivers): no hiding at all (b + small ops = b + the full small-op time). Device
  profile: on 32 cores the HiFi4 matmuls are compute-bound (~87 ns per 32x32x32 tile; qkvg 50 us even with
  the GCB pre-filled during a long gap). Ring 64 halves compute per core; then the stream is the bound and
  the prefetcher runs ahead during the small ops.
- Small in1 pages are slow for the DRISC sender (ring 32: in0_block_w 2 -> 400 us, 4 -> 302, 8 -> 273 us per
  layer, matmuls only). Use the largest in0_block_w with a page of <= 64-128 KB (ring 64: 35-98 KB pages).
- The first matmul of a frame waits ~25 us when the request is only sent at execute_trace (captured request);
  queueing frames ahead removes that (weights are constant, the stream does not depend on data).
- While the prefetcher streams, small DRAM-reading ops slow down (adds 2.4 -> 5-8 us right after a matmul).
  The stand-ins read one row; the real SDPA / KV updates read more DRAM, so expect more of this contention.

## Recipe (one p150, no sub-devices needed: the senders are the DRAM cores, not worker cores)

1. `ttnn.experimental.is_tensor_prefetcher_supported(mesh)` (programmable DRAM cores, fw >= 19.12; not with
   TT_METAL_STREAMING_PROFILER=1).
2. Receivers = matmul cores = grid 8 cols x 8 rows at (0,0)-(7,7); ring = 64. Every weight's N must be a
   multiple of 64 tiles: q|gate|k|v (288 tiles) padded with zeros to 320 tiles (+11 % bytes for that matmul).
3. Weights in receiver-contiguous DRAM layout: `MemoryConfig(BufferType.DRAM, NdShardSpec(Shape([K, N/64]),
   banks (0,0)-(7,0), ROW_MAJOR, ROUND_ROBIN_1D))`; bank b feeds ring positions b, b+8, ... (grid column b).
4. Program config per matmul: `MatmulMultiCoreReuseMultiCast1DProgramConfig(grid (8, 8), in0_block_w = divisor
   of K tiles with page <= 128 KB, per_core_N = N_tiles/64, out_block_w = per_core_N, out_block_h = per_core_M = 1,
   mcast_in0=True, gather_in0=False, num_global_cb_receivers=8, fused_activation as needed)`.
5. One GCB for all matmuls: `ttnn.experimental.create_global_circular_buffer_for_matmul_1d(mesh, configs,
   weights, bank_to_receivers, size)` (size = bytes per receiver core, >= 2 pages of every config; 512 KB-1 MB).
6. `ttnn.experimental.start_tensor_prefetcher(mesh)`; per frame `queue_tensor_prefetcher_request(mesh,
   [(w, tensor_prefetcher_block_count_for_matmul_1d(cfg, w, gcb)) for every matmul in order], global_cb=gcb,
   capture_into_trace=True)` inside the trace capture (or queue many frames ahead, not captured).
7. `ttnn.linear(x, w, program_config=cfg, global_cb=gcb, compute_kernel_config=ckc)`; x interleaved DRAM, output
   interleaved: no reshard needed. Every queued tensor must be consumed by exactly one matmul, in order
   (otherwise the stream desyncs and stop hangs). `stop_tensor_prefetcher(mesh)` before closing the device.

## Constraints found
- bf16 and bfp8 weights both work (bfp4 not tested). Mixed page sizes in one GCB work.
- GCB L1: 512 KB (or 1 MB) permanently reserved on 64 cores; the other ops' CBs must still fit (the stand-ins did).
- Weights need a second, ND-sharded copy if prefill keeps using the interleaved ones (+1.5 GB DRAM for 13 layers).
- Trace: works both ways (a captured request is re-sent by the host on every execute_trace).
- Card 0 has a 12x10 grid; the recipe only uses (0..7, 0..7), so it should map unchanged (not run on card 0).
