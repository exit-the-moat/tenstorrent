#!/bin/bash
# warm repeats: no page-cache drop between the runs (r2/r3/b1/b2 had cold-cache first-line outliers), one
# sync + drop + sha256 check over all six dirs at the end
source /opt/ep0/port7/env7.sh
BEST="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"
echo "=== chain9 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
for k in 4 5 6; do
  lock; NODROP=1 EXTRA_ENV="$BEST" $P/render7b.sh $P/r$k; unlock
  lock; NODROP=1 EXTRA_ENV="$BEST" $P/render7b.sh $P/b$k --tower-mlp bfp8; unlock
done
sync; echo 3 > /proc/sys/vm/drop_caches
for k in 4 5 6; do for x in r b; do (cd $P/$x$k && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK_DISK $x$k"); done; done
for k in 4 5 6; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r$k $P/i_et2" 2>&1 | grep "^[a-z_]* {" > $P/r$k.cmp; echo "r$k $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r$k.cmp) of 8 identical to i_et2"; done
for k in 4 5 6; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/b$k $P/i_bf8" 2>&1 | grep "^[a-z_]* {" > $P/b$k.cmp; echo "b$k $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/b$k.cmp) of 8 identical to i_bf8"; done
python3 $P/rtf.py $P/r4 $P/r5 $P/r6 $P/b4 $P/b5 $P/b6
echo "=== chain9 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN9_DONE
