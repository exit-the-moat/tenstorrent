#!/bin/bash
# Day 5 chain: golden (base, BFP8 x8), gate gens (base seed 0 = regression vs day4b/out_v2; BFP8 x8 seeds 0,1,2), Whisper
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
L8=0,27,28,30,32,33,36,37
cd /opt/ep0/day5
echo "=== chain1 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
$D/run_check.sh golden_base.log tests.check_paged_backbone --out /opt/ep0/day5/golden_base.json; tail -1 golden_base.log
EXTRA_ENV="QWEN36_EXPERT_GU_BFP8_LAYERS=$L8" $D/run_check.sh golden_bfp8_8.log tests.check_paged_backbone --out /opt/ep0/day5/golden_bfp8_8.json; tail -1 golden_bfp8_8.log
$D/run_gen.sh gate_base.log --fast --lines close,fix --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5/gate_base; tail -1 gate_base.log
EXTRA_ENV="QWEN36_EXPERT_GU_BFP8_LAYERS=$L8" $D/run_gen.sh gate_bfp8_8.log --fast --lines close,fix --seeds 0,1,2 --ref /opt/ep0/voice/ref_v2.wav --out /opt/ep0/day5/gate_bfp8_8; tail -1 gate_bfp8_8.log
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate_seeds /opt/ep0/day5/gate_bfp8_8" > whisper_bfp8_8.log 2>&1
echo "=== chain1 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN1_DONE
