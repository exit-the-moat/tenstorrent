#!/bin/bash
source /opt/ep0/port7/env7.sh
BEST="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"
until grep -q CHAIN7_DONE $P/chain7.log; do sleep 15; done
echo "=== chain8 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
for k in 1 2 3; do
  lock; EXTRA_ENV="$BEST" $P/render7.sh $P/r$k; unlock
  lock; EXTRA_ENV="$BEST" $P/render7.sh $P/b$k --tower-mlp bfp8; unlock
done
lock; $D/run_gen.sh $P/r_def.log --fast --lines all --ref /opt/ep0/voice/ref_v2.wav --out $P/r_def; echo "r_def $(tail -1 $P/r_def.log)"; unlock
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r_def /opt/ep0/day4b/out_v2" 2>&1 | grep "^[a-z_]* {" > $P/r_def.cmp
echo "r_def $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r_def.cmp) of 8 identical to day4b/out_v2"
for k in 1 2 3; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r$k $P/i_et2" 2>&1 | grep "^[a-z_]* {" > $P/r$k.cmp; echo "r$k $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r$k.cmp) of 8 identical to i_et2"; done
for k in 1 2 3; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/b$k $P/i_bf8" 2>&1 | grep "^[a-z_]* {" > $P/b$k.cmp; echo "b$k $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/b$k.cmp) of 8 identical to i_bf8"; done
python3 $P/rtf.py $P/r1 $P/r2 $P/r3 $P/b1 $P/b2 $P/b3
echo "=== chain8 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN8_DONE
