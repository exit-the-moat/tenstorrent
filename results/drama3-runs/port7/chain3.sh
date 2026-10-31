#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "=== chain3 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
$D/run_gen.sh $P/i_base7.log --interactive --lines all $R --out $P/i_base7
echo "i_base7 $(tail -1 $P/i_base7.log) $(date +%T)"
unlock; lock
EXTRA_ENV="DRAMA3_FRAME_PIPE=1" $D/run_gen.sh $P/i_pipe.log --interactive --lines all $R --out $P/i_pipe
echo "i_pipe $(tail -1 $P/i_pipe.log) $(date +%T)"
unlock
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_base7 /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_base7.cmp
echo "i_base7 $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_base7.cmp) of 8 identical to i_qdt"
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_pipe /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_pipe.cmp
echo "i_pipe $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_pipe.cmp) of 8 identical to i_qdt"
echo "=== chain3 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN3_DONE
