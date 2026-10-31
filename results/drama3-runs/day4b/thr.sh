#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
for t in 8 10; do
$D/run_gen.sh /opt/ep0/day4b/thr$t.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --codec-threads $t --out /opt/ep0/day4b/thr$t
done
echo THR_DONE
