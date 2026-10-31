#!/bin/bash
# Day 5b measurements: the 8 test lines (ref_v2, seed 0) at batch 2 / 4 / 8, card power sampled alongside
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
cd /opt/ep0/day5b
echo "=== t3 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
for B in ${SIZES:-8 4 2}; do
  out=/opt/ep0/day5b/m$B; mkdir -p $out; rm -f $out/power.stop
  python3 $D/power_log.py log $out/power.jsonl $out/power.stop &
  $D/run_gen.sh $out/gen.log --batch --batch-size $B --lines all --ref /opt/ep0/voice/ref_v2.wav --out $out
  touch $out/power.stop; wait
  python3 $D/power_log.py merge $out/power.jsonl $out/timing.json
  /opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_batch $out" > $out/summary.log 2>&1
  echo "B=$B $(tail -1 $out/gen.log) $(tail -1 $out/summary.log)"
done
echo "=== t3 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo T3_DONE
