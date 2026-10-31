#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
CMP="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1 QWEN36_GROUPED_COMPACT=1"
$P/gate.sh c_cmp "$CMP" 2>&1 | sed "s/^/gate: /"
for k in 2 3; do lock; NODROP=1 EXTRA_ENV="$CMP" $P/render7b.sh $P/c_cmp$k; unlock; done
lock; NODROP=1 EXTRA_ENV="$CMP" $P/render7b.sh $P/c_cmpb --tower-mlp bfp8; unlock
sync; echo 3 > /proc/sys/vm/drop_caches
for x in c_cmp2 c_cmp3 c_cmpb; do (cd $P/$x && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK_DISK $x"); done
for x in c_cmp2 c_cmp3; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/$x $P/c_cmp" 2>&1 | grep "^[a-z_]* {" > $P/$x.cmp; echo "$x $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/$x.cmp) of 8 identical to c_cmp"; done
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $P/c_cmpb" > $P/c_cmpb.wcheck.log 2>&1; echo "wcheck $(grep "mean match" $P/c_cmpb.wcheck.log)"
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-spk:\$PYTHONPATH python /opt/ep0/port7/spk.py /opt/ep0/port6/i_qdt $P/c_cmp $P/c_cmpb $P/c_ds $P/c_dsb" > $P/spk2.log 2>&1; grep "^{" $P/spk2.log
python3 $P/rtf.py $P/c_cmp $P/c_cmp2 $P/c_cmp3 $P/c_cmpb
echo "io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN12_DONE
