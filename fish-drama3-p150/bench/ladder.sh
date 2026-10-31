#!/bin/bash
# Parallel voices, the port6 ladder: all port6 batch levers (v4 GDN, batch MoE, batch attention) + matmul glue + quantizer on the card
# c<B>: continuous batching (host.cbatch, a new caller every 1.0 s, each line twice); s<B>: static batch (run_gen --batch).
# Card power is logged per run (power_log.py) and summed into timing.json (power_sum.py). Usage: bench/ladder.sh [out dir]
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/env.sh
P=${1:-/opt/ep0/ladder}
mkdir -p $P
R="--ref $REF"
B7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1"
pw_start() { rm -f $1/power.stop; python3 $D/power_log.py log $1/power.jsonl $1/power.stop & }
pw_stop() { touch $1/power.stop; sleep 3; python3 $HERE/power_sum.py $1/power.jsonl $1/timing.json > $1/power.txt 2>&1; }
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
for spec in c2:2:0.8:0.3:guard:0.5 c3:3:0.9:0.4:guard:0.5 c4:4:0.97:0.5:guard:0.6 c4f:4:none:0:warm-first c8f:8:none:0:warm-first; do
  IFS=: read name B r pre pol gv <<< "$spec"
  [ -n "$gv" ] && pol="$pol:$gv"
  ramp=""; [ "$r" != none ] && ramp="DRAMA3_STREAM_RAMP=$r:$pre"
  o=$P/$name; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7 $ramp" $D/run_check.sh $o/gen.log host.cbatch --batch-size $B --lines all --repeat 2 --arrivals every:1.0 --policy $pol $R --out $o
  pw_stop $o; echo "$name $(tail -1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
done
for B in 2 3 4 8; do
  o=$P/s$B; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7" $D/run_gen.sh $o/gen.log --batch --batch-size $B --lines all $R --out $o
  pw_stop $o; echo "s$B $(tail -1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
done
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
echo DONE
