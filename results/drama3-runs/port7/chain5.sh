#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "=== chain5 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
EXTRA_ENV="$I7" $D/run_check.sh $P/tower_base.log tests.check_tower_device; echo "tower_base $(tail -1 $P/tower_base.log)"
unlock; lock
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1" $D/run_check.sh $P/tower_et.log tests.check_tower_device; echo "tower_et $(tail -1 $P/tower_et.log)"
unlock; lock
EXTRA_ENV="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1" $D/run_gen.sh $P/i_lean.log --interactive --lines all $R --out $P/i_lean; echo "i_lean $(tail -1 $P/i_lean.log) $(date +%T)"
unlock; lock
EXTRA_ENV="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1" $D/run_gen.sh $P/i_et.log --interactive --lines all $R --out $P/i_et; echo "i_et $(tail -1 $P/i_et.log) $(date +%T)"
unlock
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_lean /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_lean.cmp
echo "i_lean $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_lean.cmp) of 8 identical to i_qdt"
echo "=== chain5 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN5_DONE
