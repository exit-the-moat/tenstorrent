#!/bin/bash
# port8 item 4: tune the BFP4 tower MLP matmuls, render with the copied BFP8 configs (s8b4c), floor numbers in the write-up
source /opt/ep0/port8/env8.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
S8="$P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8"
lock
NODROP=1 EXTRA_ENV="$S8" timeout 900 $P/render8.sh $P/s8b4c --tower-mlp bfp4
EXTRA_ENV="$I7 TUNE_TOWER_MLP=bfp4 DRAMA3_DECODE_LINEAR=0" timeout 1500 $D/run_check.sh $P/tune_b4.log tests.tune_decode_linear --out $P/tune_b4.json; echo "tune_b4 $(tail -1 $P/tune_b4.log) $(date +%T)"
unlock
python3 /opt/ep0/port7/rtf.py $P/s8_1 $P/s8b4 $P/s8b4c $P/s8b8
echo "io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN7_DONE
