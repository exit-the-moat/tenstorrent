#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
cd /opt/ep0/day4b
./run_check.sh check_prefix2.log tests.check_prefix_cache --lines close,fix --out /opt/ep0/day4b/check_prefix2.json
EXTRA_ENV="QWEN36_GROUPED_BMM_CONFIG=0" $D/run_gen.sh /opt/ep0/day4b/gate2c.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day4b/gate2c
echo G5_DONE
