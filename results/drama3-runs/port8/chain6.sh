#!/bin/bash
# port8: gates of the renders, speaker / spectra, defaults bit-identical, ExpertMM no-math profile, stretch c2 / c4, smoke
source /opt/ep0/port8/env8.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
S8="$P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8"
$P/gate8.sh s8_1 "$S8" "" 2>&1 | sed "s/^/gate s8_1: /"
$P/gate8.sh s8b4 "$S8" "--tower-mlp bfp4" 2>&1 | sed "s/^/gate s8b4: /"
$P/gate8.sh s8b4a "$S8 DRAMA3_TOWER_ATTN_BFP4=1" "--tower-mlp bfp4" 2>&1 | sed "s/^/gate s8b4a: /"
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $P/s8b8" > $P/s8b8.wcheck.log 2>&1; echo "wcheck s8b8 $(grep "mean match" $P/s8b8.wcheck.log)"
for x in s8_2 s8_3; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/$x $P/s8_1" 2>&1 | grep "^[a-z_]* {" > $P/$x.cmp; echo "$x $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/$x.cmp) of 8 identical to s8_1"; done
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-spk:\$PYTHONPATH python /opt/ep0/port7/spk.py /opt/ep0/port6/i_qdt $P/s8_1 $P/s8b8 $P/s8b4 $P/s8b4a" > $P/spk.log 2>&1; grep "^{" $P/spk.log | cut -c1-300
/opt/ep0/run.sh "python /opt/ep0/port7/spec_cmp.py /opt/ep0/port6/i_qdt $P/s8_1 $P/s8b8 $P/s8b4 $P/s8b4a" > $P/spec_cmp.log 2>&1; grep "^{" $P/spec_cmp.log | cut -c1-400
lock
$D/run_gen.sh $P/r_def.log --fast --lines all $R --out $P/r_def
$D/run_gen.sh $P/i_base.log --interactive --lines all $R --out $P/i_base
EXTRA_ENV="$I7 $S8 DRAMA3_MOE_EXPERT_MM_NO_MATH=1 TT_METAL_DEVICE_PROFILER=1" timeout 600 $D/run_check.sh $P/prof_nomath.log tests.profile_frame7 --mode prof --out $P/prof_nomath.json; echo "prof_nomath $(tail -1 $P/prof_nomath.log)"
unlock
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r_def /opt/ep0/day4b/out_v2" 2>&1 | grep "^[a-z_]* {" > $P/r_def.cmp; echo "r_def $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r_def.cmp) of 8 identical to day4b/out_v2"
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_base /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_base.cmp; echo "i_base $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_base.cmp) of 8 identical to i_qdt"
# stretch: 2 and 4 concurrent callers, port6 lad3 c2 / c4 settings (port8 levers are single-row: not active in batch mode)
B7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1"
pw_start() { rm -f $1/power.stop; python3 $D/power_log.py log $1/power.jsonl $1/power.stop & }
pw_stop() { touch $1/power.stop; sleep 3; python3 /opt/ep0/port6/power_sum.py $1/power.jsonl $1/timing.json > $1/power.txt 2>&1; }
lock
for spec in c2:2:0.8:0.3:guard:0.5 c4:4:0.97:0.5:guard:0.6; do
  IFS=: read name B r pre pol gv <<< "$spec"
  pol="$pol:$gv"; ramp="DRAMA3_STREAM_RAMP=$r:$pre"
  o=$P/st_$name; mkdir -p $o; pw_start $o
  EXTRA_ENV="$B7 $ramp" timeout 900 $D/run_check.sh $o/gen.log host.cbatch --batch-size $B --lines all --repeat 2 --arrivals every:1.0 --policy $pol $R --out $o
  pw_stop $o; echo "st_$name $(tail -1 $o/gen.log) $(cat $o/power.txt) $(date +%T)"
done
/opt/ep0/run.sh "python /opt/ep0/smoke.py" > $P/smoke_end.log 2>&1; grep SMOKE $P/smoke_end.log
unlock
echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN6_DONE
