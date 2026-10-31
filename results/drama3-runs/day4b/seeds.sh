#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
EXTRA_ENV="DRAMA3_PREFIX_CACHE=0 QWEN36_GROUPED_BMM_CONFIG=0" $D/run_gen.sh /opt/ep0/day4b/seeds_base.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --seeds 1,2,3,4,5 --out /opt/ep0/day4b/seeds_base
$D/run_gen.sh /opt/ep0/day4b/seeds_new.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --seeds 1,2,3,4,5 --out /opt/ep0/day4b/seeds_new
echo SEEDS_DONE
