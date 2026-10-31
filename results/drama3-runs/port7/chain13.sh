#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
lock; $D/run_gen.sh $P/r_def2.log --fast --lines all $R --out $P/r_def2; unlock
lock; $D/run_gen.sh $P/i_base8.log --interactive --lines all $R --out $P/i_base8; unlock
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r_def2 /opt/ep0/day4b/out_v2" 2>&1 | grep "^[a-z_]* {" > $P/r_def2.cmp
echo "r_def2 $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r_def2.cmp) of 8 identical to day4b/out_v2"
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_base8 /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_base8.cmp
echo "i_base8 $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_base8.cmp) of 8 identical to i_qdt"
lock; /opt/ep0/run.sh "python /opt/ep0/smoke.py" > $P/smoke3.log 2>&1; unlock; grep SMOKE $P/smoke3.log
echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN13_DONE
