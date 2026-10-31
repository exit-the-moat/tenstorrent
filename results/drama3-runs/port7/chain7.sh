#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
BEST="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"
echo "=== chain7 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
/opt/ep0/run.sh "python -m models.demos.blackhole.drama3.tests.bench_prefill_mm" > $P/bench_prefill_mm.log 2>&1; echo "prefill_mm EXIT $? $(date +%T)"
unlock; lock
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1" $D/run_check.sh $P/tower_et_bf8.log tests.check_tower_device --mlp bfp8; echo "tower_et_bf8 $(tail -1 $P/tower_et_bf8.log)"
unlock; lock
EXTRA_ENV="$BEST" $D/run_gen.sh $P/i_bf8.log --interactive --lines all $R --out $P/i_bf8 --tower-mlp bfp8; echo "i_bf8 $(tail -1 $P/i_bf8.log) $(date +%T)"
unlock
$P/gate.sh i_bf8 "$BEST" "--tower-mlp bfp8" 2>&1 | sed "s/^/gate: /"
echo "=== chain7 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN7_DONE
