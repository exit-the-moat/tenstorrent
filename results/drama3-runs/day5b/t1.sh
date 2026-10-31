#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
cd /opt/ep0/day5b
$D/run_gen.sh b2_close.log --batch --batch-size 2 --lines close --seeds 0,1 --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5b/b2_close
/opt/ep0/run.sh "python /opt/ep0/day5/compare_frames.py /opt/ep0/day5b/b2_close/close_s0.frames.pt /opt/ep0/day4b/out_v2/close.frames.pt /opt/ep0/day5b/b2_close/close_s1.frames.pt /opt/ep0/day4b/seeds_final/close_s1.frames.pt" 2>&1 | grep -E "IDENT|prefix"
echo T1_DONE
