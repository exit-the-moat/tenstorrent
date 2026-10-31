#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
$D/run_gen.sh /opt/ep0/day4b/gate2b.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day4b/gate2b
EXTRA_ENV="DRAMA3_TOWER_DEVICE_SAMPLING=1" $D/run_gen.sh /opt/ep0/day4b/gate3.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day4b/gate3
echo G4_DONE
