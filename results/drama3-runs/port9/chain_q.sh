#!/bin/bash
# port9 part 2: full queue (32 jobs at t=0, greedy) at B = 16 / 12 / 8, short test lines and ~30-45 s narration chunks
source /opt/ep0/port8/env8.sh
P=/opt/ep0/port9
R="--ref /opt/ep0/voice/ref_v2.wav"
lock() { while [ -e /opt/ep0/card.wanted ]; do sleep 30; done; until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo ep0-port9 > /opt/ep0/card.lock/owner; }
B7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1"
pw_start() { rm -f $1/power.stop; python3 $D/power_log.py log $1/power.jsonl $1/power.stop & }
pw_stop() { touch $1/power.stop; sleep 3; python3 /opt/ep0/port6/power_sum.py $1/power.jsonl $1/timing.json > $1/power.txt 2>&1; }
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
lock
for N in 16 12 8; do
  o=$P/q_short_$N; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7" timeout 1200 $D/run_check.sh $o/gen.log host.cbatch --batch-size $N --lines all --repeat 4 --arrivals 0 --policy greedy $R --out $o
  pw_stop $o; echo "q_short_$N $(tail -n 1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
  grep -q "EXIT 0" $o/gen.log || { grep -iE "error|assert|Traceback" $o/gen.log | tail -n 3 | sed "s/^/  q_short_$N fail: /"; }
done
for N in 16 12 8; do
  o=$P/q_long_$N; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7" timeout 1800 $D/run_check.sh $o/gen.log host.cbatch --batch-size $N --lines-file $P/long_lines.json --lines all --repeat 4 --max-frames 1200 --arrivals 0 --policy greedy $R --out $o
  pw_stop $o; echo "q_long_$N $(tail -n 1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
  grep -q "EXIT 0" $o/gen.log || { grep -iE "error|assert|Traceback" $o/gen.log | tail -n 3 | sed "s/^/  q_long_$N fail: /"; }
done
/opt/ep0/run.sh "python /opt/ep0/smoke.py" > $P/smoke_q.log 2>&1; grep SMOKE $P/smoke_q.log
rm -rf /opt/ep0/card.lock
for N in 16 12 8; do
  [ -f $P/q_short_$N/timing.json ] && { /opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $P/q_short_$N" > $P/q_short_$N.wcheck.log 2>&1; echo "wcheck q_short_$N $(grep "mean match" $P/q_short_$N.wcheck.log)"; }
  [ -f $P/q_long_$N/timing.json ] && { /opt/ep0/run.sh "LINES_FILE=$P/long_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python $P/wcheck_lf.py $P/q_long_$N" > $P/q_long_$N.wcheck.log 2>&1; echo "wcheck q_long_$N $(grep "mean match" $P/q_long_$N.wcheck.log)"; }
done
echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN_Q_DONE
