#!/bin/bash
# Episode 0: run the generator in the TT container (backbone prefill HiFi2).
#   [DECODE_FIDELITY=HiFi2] run_gen.sh <log file> [--fast] <generator args...>
#   --fast: host/generate_fast.py (Day 4: paged + traced backbone, tower on the card, streaming codec) with the
#           Day 4 decode settings (expert decode matmuls: fp32-accumulating LoFi, K blocks 32 / 8) and the Day 4b
#           indexed top-k MoE decode (MOE_INDEXED=0 for the dense 256-slot path), router top-k on the raw scores
#           (ROUTER_TOPK_FIRST=0: softmax over all 256 first), the lean GDN decode step (GDN_LEAN=0: original),
#           the one-token GDN gate without the never-binding clips (GDN_NOCLIP=0), the grouped-MoE gate_up
#           on an 8x8 grid from M=256 (GU_GRID_MIN_M=0: default program), q norm + scale in one multiply (GDN_QSCALE=0);
#   --interactive: --fast plus the port5 low-latency levers for live use (Sebastian, 2026-10-30): first stream window
#           after 4 frames, decode the first window right away, voice prefix prefilled at startup, upstream fused GDN
#           chunk prefill, small-grid grouped-MoE suffix prefill, and every port5 fused decode kernel that passed the
#           golden + close/fix gates (GDN v3, MoE top-k / combine / SwiGLU, attention pre / post / merged qkv).
#           port6 (2026-10-30): grouped-MoE matmul glue in the suffix prefill, codec quantizer decode on the card
#           (gapless playback from the first sound).
#           port10 (2026-10-31, Sebastian): codec decoder traced, channel-0 readback, causal padding, fused snake
#           kernel (audio bit-identical; first window 31 -> 15 ms, RTF 0.537 -> 0.518).
#           port10 (2026-10-31, Sebastian: "all improvements"): + the port7 / port8 same-precision levers (frame pipe,
#           lean sampler, e tables, compact suffix prefill, fused add + norm, top-k + softmax, ExpertMM v2) and the shared
#           expert inside ExpertMM. Changes the takes slightly (gated in port7 / port8 / port10); reference = port10/i_all.
#           + --tower-mlp bfp8 (Sebastian, 2026-10-31: sounds identical; RTF 0.463 -> 0.440). Not included: FIRST_WINDOW=2.
#           + the fused codebook sampler on 4 cores (bit-identical to port10/i_bf8; RTF ~0.433 -> ~0.407).
#           + backbone + tower DRAM weight prefetch (Sebastian, by ear: good; card 0 RTF 0.412 -> 0.381; +2.56 GB DRAM;
#           single stream only). Reference = port10/k_on_1.
#           Not included (quality calls): bfp8 tower MLP, FIRST_WINDOW=2. Episode renders keep --fast (bit-identical to day4b/out_v2).
#   else host/generate.py (Day 3).
log=$1; shift
mod=generate
extra=""
inter=""
if [ "${1:-}" = "--interactive" ]; then
  inter="DRAMA3_FIRST_WINDOW=4 DRAMA3_FIRST_WINDOW_WAIT=1 DRAMA3_PREWARM_VOICE=1 QWEN36_GDN_FUSED_PREFILL=1 QWEN36_GROUPED_SMALL_GRID=1 DRAMA3_GDN_FUSED_DECODE=all DRAMA3_GDN_FUSED_V=3 DRAMA3_MOE_FUSED_TOPK=1 DRAMA3_MOE_FUSED_COMBINE=1 DRAMA3_MOE_FUSED_SWIGLU=1 DRAMA3_ATTN_FUSED_PRE=1 DRAMA3_ATTN_FUSED_POST=1 DRAMA3_ATTN_QKV_MERGED=1 QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1 DRAMA3_CODEC_DEV_TRACE=1 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1 DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1 QWEN36_GROUPED_COMPACT=1 DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8 DRAMA3_MOE_SHARED_IN_EMM=1 DRAMA3_SAMPLER_FUSED=1 DRAMA3_SAMPLER_FUSED_CORES=4 DRAMA3_BB_PREFETCH=1 DRAMA3_TOWER_PREFETCH=1"
  set -- --fast --tower-mlp bfp8 "${@:2}"  # port10: bfp8 tower MLP (Sebastian, by ear: identical); a later --tower-mlp wins
fi
if [ "${1:-}" = "--fast" ] || [ "${1:-}" = "--batch" ]; then
  # --batch: host/batch.py (Day 5b, B streams at once) with the same env
  mod=generate_fast; [ "$1" = "--batch" ] && mod=batch; shift
  extra="QWEN36_DECODE_MATH_FIDELITY=${DECODE_FIDELITY:-LoFi} QWEN36_MOE_DECODE_BLOCK_GATE_UP=${MOE_BLOCK_GU:-32} QWEN36_MOE_DECODE_BLOCK_DOWN=${MOE_BLOCK_DN:-8} QWEN36_MOE_DECODE_INDEXED=${MOE_INDEXED:-1} QWEN36_ROUTER_TOPK_FIRST=${ROUTER_TOPK_FIRST:-1} QWEN36_GDN_LEAN_DECODE=${GDN_LEAN:-1} QWEN36_GDN_DECODE_NOCLIP=${GDN_NOCLIP:-1} QWEN36_GROUPED_GU_GRID_MIN_M=${GU_GRID_MIN_M:-256} QWEN36_GDN_LEAN_QSCALE=${GDN_QSCALE:-1} $inter"
elif [ -n "${DECODE_FIDELITY:-}" ]; then
  extra="QWEN36_DECODE_MATH_FIDELITY=$DECODE_FIDELITY"
fi
/opt/ep0/run.sh "PYTHONPATH=\$PYTHONPATH:/opt/ep0/pyext-codec TT_METAL_CACHE=/opt/ep0/jit_cache \
QWEN36_MOE_MATH_FIDELITY=HiFi2 QWEN36_MLP_MATH_FIDELITY=HiFi2 QWEN36_ATTN_MATH_FIDELITY=HiFi2 $extra ${EXTRA_ENV:-} \
python -m models.demos.blackhole.drama3.host.$mod $*" > "$log" 2>&1
echo "EXIT $?" >> "$log"
