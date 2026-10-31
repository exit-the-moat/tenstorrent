#!/bin/bash
# port7 chain1: DRAM matmul bench, frame time, frame device profile (interactive env)
source /opt/ep0/port6/env.sh
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
P=/opt/ep0/port7
lock() { until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo ep0-port7 > /opt/ep0/card.lock/owner; }
unlock() { rm -rf /opt/ep0/card.lock; }
waitwanted() { while [ -e /opt/ep0/card.wanted ]; do sleep 30; done; }
echo "=== chain1 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
waitwanted; lock
/opt/ep0/run.sh "python -m models.demos.blackhole.drama3.tests.bench_dram_mm --out $P/bench_dram_mm.json" > $P/bench_dram_mm.log 2>&1; echo "bench EXIT $? $(date +%T)"
unlock; waitwanted; lock
EXTRA_ENV="$INTER" $D/run_check.sh $P/frame_time.log tests.profile_frame7 --mode time --out $P/frame_time.json; echo "time $(tail -1 $P/frame_time.log) $(date +%T)"
unlock; waitwanted; lock
EXTRA_ENV="$INTER TT_METAL_DEVICE_PROFILER=1" $D/run_check.sh $P/frame_prof.log tests.profile_frame7 --mode prof --out $P/frame_prof.json; echo "prof $(tail -1 $P/frame_prof.log) $(date +%T)"
unlock
echo "=== chain1 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN1_DONE
