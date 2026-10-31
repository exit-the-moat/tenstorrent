#!/bin/bash
source /opt/ep0/port7/env7.sh
echo "=== chain2 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
/opt/ep0/run.sh "python -m models.demos.blackhole.drama3.tests.bench_sample_ops" > $P/bench_sample_ops.log 2>&1; echo "ops EXIT $? $(date +%T)"
unlock; lock
EXTRA_ENV="$I7" $D/run_check.sh $P/kept.log tests.count_kept --kept-out $P/kept.json --lines all --ref /opt/ep0/voice/ref_v2.wav --out $P/kept_out --seeds 0,1,2; echo "kept $(tail -1 $P/kept.log) $(date +%T)"
unlock
echo "=== chain2 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN2_DONE
