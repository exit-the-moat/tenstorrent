#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "=== chain6 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
for v in "i_pipe2:DRAMA3_FRAME_PIPE=1" "i_lean2:DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1" "i_et2:DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"; do
  n=${v%%:*}; e=${v#*:}
  lock
  EXTRA_ENV="$e" $D/run_gen.sh $P/$n.log --interactive --lines all $R --out $P/$n; echo "$n $(tail -1 $P/$n.log) $(date +%T)"
  unlock
  /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/$n /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/$n.cmp
  echo "$n $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/$n.cmp) of 8 identical to i_qdt"
done
echo "=== chain6 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN6_DONE
