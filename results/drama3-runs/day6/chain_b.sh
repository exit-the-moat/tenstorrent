#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
$D/render_day6_batch.sh 8 /opt/ep0/day6/narration_b8
tail -1 /opt/ep0/day6/narration_b8/summary.log
# final regression of the default pipeline after all Day 5b patches
$D/run_gen.sh /opt/ep0/day5b/final_default.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5b/final_default
/opt/ep0/run.sh "python /opt/ep0/day5/compare_frames.py /opt/ep0/day5b/final_default/close.frames.pt /opt/ep0/day4b/out_v2/close.frames.pt /opt/ep0/day5b/final_default/fix.frames.pt /opt/ep0/day4b/out_v2/fix.frames.pt" 2>&1 | grep -E "IDENT|prefix"
cmp /opt/ep0/day5b/final_default/close.wav /opt/ep0/day4b/out_v2/close.wav && cmp /opt/ep0/day5b/final_default/fix.wav /opt/ep0/day4b/out_v2/fix.wav && echo WAVS_IDENTICAL
echo CHAIN_B_DONE
