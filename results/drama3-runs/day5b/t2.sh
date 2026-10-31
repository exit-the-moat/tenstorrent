#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
C="/opt/ep0/run.sh python /opt/ep0/day5/compare_frames.py"
cd /opt/ep0/day5b
echo "=== t2 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
$D/run_gen.sh b1.log --batch --batch-size 1 --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5b/b1; tail -1 b1.log
$C /opt/ep0/day5b/b1/close.frames.pt /opt/ep0/day4b/out_v2/close.frames.pt /opt/ep0/day5b/b1/fix.frames.pt /opt/ep0/day4b/out_v2/fix.frames.pt 2>&1 | grep -E "IDENT|prefix"
$D/run_gen.sh b2.log --batch --batch-size 2 --lines close --seeds 0,1 --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5b/b2; tail -1 b2.log
$C /opt/ep0/day5b/b2/close_s0.frames.pt /opt/ep0/day4b/out_v2/close.frames.pt /opt/ep0/day5b/b2/close_s1.frames.pt /opt/ep0/day4b/seeds_final/close_s1.frames.pt 2>&1 | grep -E "IDENT|prefix"
$D/run_gen.sh b4.log --batch --batch-size 4 --lines close --seeds 1,2,3,4 --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5b/b4; tail -1 b4.log
for s in 1 2 3 4; do $C /opt/ep0/day5b/b4/close_s$s.frames.pt /opt/ep0/day4b/seeds_final/close_s$s.frames.pt 2>&1 | grep -E "IDENT|prefix"; done
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate_seeds /opt/ep0/day5b/b2 /opt/ep0/day5b/b4" 2>&1 | grep "^{"
grep -E "batch [0-9]+:" b1.log b2.log b4.log
echo "=== t2 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo T2_DONE
