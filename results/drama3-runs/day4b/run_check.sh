#!/bin/bash
# day4b: run a drama3 test module with the --fast env. run_check.sh <log> <module> <args...>
log=$1; shift; mod=$1; shift
/opt/ep0/run.sh "PYTHONPATH=\$PYTHONPATH:/opt/ep0/pyext-codec TT_METAL_CACHE=/opt/ep0/jit_cache \
QWEN36_MOE_MATH_FIDELITY=HiFi2 QWEN36_MLP_MATH_FIDELITY=HiFi2 QWEN36_ATTN_MATH_FIDELITY=HiFi2 \
QWEN36_DECODE_MATH_FIDELITY=LoFi QWEN36_MOE_DECODE_BLOCK_GATE_UP=32 QWEN36_MOE_DECODE_BLOCK_DOWN=8 QWEN36_MOE_DECODE_INDEXED=1 QWEN36_ROUTER_TOPK_FIRST=1 QWEN36_GDN_LEAN_DECODE=1 QWEN36_GDN_DECODE_NOCLIP=1 QWEN36_GROUPED_GU_GRID_MIN_M=256 QWEN36_GDN_LEAN_QSCALE=1 ${EXTRA_ENV:-} \
python -m models.demos.blackhole.drama3.$mod $*" > "$log" 2>&1
echo "EXIT $?" >> "$log"
