#!/bin/bash
# port9: 5-8 concurrent streams (Sebastian 2026-10-31). Same batch flags as port8 chain6 stretch.
source /opt/ep0/port8/env8.sh
P=/opt/ep0/port9
R="--ref /opt/ep0/voice/ref_v2.wav"
lock() { while [ -e /opt/ep0/card.wanted ]; do sleep 30; done; until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo ep0-port9 > /opt/ep0/card.lock/owner; }
B7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1"
pw_start() { rm -f $1/power.stop; python3 $D/power_log.py log $1/power.jsonl $1/power.stop & }
pw_stop() { touch $1/power.stop; sleep 3; python3 /opt/ep0/port6/power_sum.py $1/power.jsonl $1/timing.json > $1/power.txt 2>&1; }
ALL=price,model,close,specs,cold_open,server,second_opinion,fix
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
lock
for N in 5 6 7 8; do
  # burst: N lines at t=0, greedy admission (RTX reference: 8 concurrent requests)
  o=$P/burst_$N; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7" timeout 900 $D/run_check.sh $o/gen.log host.cbatch --batch-size $N --lines $(echo $ALL | cut -d, -f1-$N) --arrivals 0 --policy greedy $R --out $o
  pw_stop $o; echo "burst_$N $(tail -n 1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
  # ladder: 16 callers, one per 1.0 s, warm-first (port6 lad3 c4f / c8f)
  o=$P/lad_$N; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7" timeout 900 $D/run_check.sh $o/gen.log host.cbatch --batch-size $N --lines all --repeat 2 --arrivals every:1.0 --policy warm-first $R --out $o
  pw_stop $o; echo "lad_$N $(tail -n 1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
done
/opt/ep0/run.sh "python /opt/ep0/smoke.py" > $P/smoke_end.log 2>&1; grep SMOKE $P/smoke_end.log
rm -rf /opt/ep0/card.lock
for N in 5 6 7 8; do /opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $P/burst_$N" > $P/burst_$N.wcheck.log 2>&1; echo "wcheck burst_$N $(grep "mean match" $P/burst_$N.wcheck.log)"; done
echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN_ST_DONE
