#!/bin/bash
# Episode 0 Day 2: all-layer chained check on the card, once per candidate-LM-head dtype.
#   [EXTRA_ENV="VAR=x ..."] run_chain.sh <out dir> [head dtypes...]   e.g. run_chain.sh /opt/ep0/day2/v3 bf16 bfp8
out=$1; shift
for h in "${@:-bf16}"; do
  /opt/ep0/run.sh "${EXTRA_ENV:-} DRAMA3_HEAD_DTYPE=$h TT_METAL_CACHE=/opt/ep0/jit_cache HF_MODEL=/opt/ep0/drama3_shim \
TT_CACHE_PATH=/opt/ep0/cache/drama3 python -m models.demos.blackhole.drama3.tt.device_check --chain --out $out" \
    > "$out.$h.log" 2>&1
done
echo ALL_DONE >> "$out.$h.log"
