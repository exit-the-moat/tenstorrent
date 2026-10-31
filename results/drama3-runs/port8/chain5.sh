#!/bin/bash
# port8: device profiles (expert matmul GB/s), tower checks for bfp4, then 8-line renders + gates of the port8 stack
source /opt/ep0/port8/env8.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
S8="$P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8"
SM="$P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1"
lock
EXTRA_ENV="$I7 $SM TT_METAL_DEVICE_PROFILER=1" timeout 600 $D/run_check.sh $P/prof_sm.log tests.profile_frame7 --mode prof --out $P/prof_sm.json; echo "prof_sm $(tail -1 $P/prof_sm.log) $(date +%T)"
EXTRA_ENV="$I7 $S8 TT_METAL_DEVICE_PROFILER=1" timeout 600 $D/run_check.sh $P/prof_s8.log tests.profile_frame7 --mode prof --out $P/prof_s8.json; echo "prof_s8 $(tail -1 $P/prof_s8.log) $(date +%T)"
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_ADDNORM_FUSED=1" timeout 600 $D/run_check.sh $P/tower_b4.log tests.check_tower_device --mlp bfp4 --out $P/tower_b4.json; echo "tower_b4 $(tail -1 $P/tower_b4.log)"
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_ADDNORM_FUSED=1 DRAMA3_TOWER_ATTN_BFP4=1" timeout 600 $D/run_check.sh $P/tower_b4a.log tests.check_tower_device --mlp bfp4 --out $P/tower_b4a.json; echo "tower_b4a $(tail -1 $P/tower_b4a.log)"
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_ADDNORM_FUSED=1" timeout 600 $D/run_check.sh $P/tower_b8.log tests.check_tower_device --mlp bfp8 --out $P/tower_b8.json; echo "tower_b8 $(tail -1 $P/tower_b8.log)"
for k in 1 2 3; do NODROP=1 EXTRA_ENV="$S8" timeout 900 $P/render8.sh $P/s8_$k; done
NODROP=1 EXTRA_ENV="$S8" timeout 900 $P/render8.sh $P/s8b8 --tower-mlp bfp8
NODROP=1 EXTRA_ENV="$S8" timeout 900 $P/render8.sh $P/s8b4 --tower-mlp bfp4
NODROP=1 EXTRA_ENV="$S8 DRAMA3_TOWER_ATTN_BFP4=1" timeout 900 $P/render8.sh $P/s8b4a --tower-mlp bfp4
unlock
python3 $P/../port7/rtf.py $P/s8_1 $P/s8_2 $P/s8_3 $P/s8b8 $P/s8b4 $P/s8b4a
echo "io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN5_DONE
