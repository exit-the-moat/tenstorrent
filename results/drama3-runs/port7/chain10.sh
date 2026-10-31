#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
BEST="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"
echo "=== chain10 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_TOWER_DS=1" $D/run_check.sh $P/tower_ds.log tests.check_tower_device; echo "tower_ds $(tail -1 $P/tower_ds.log)"
unlock; lock
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_TOWER_DS=1" $D/run_check.sh $P/tower_dsb.log tests.check_tower_device --mlp bfp8; echo "tower_dsb $(tail -1 $P/tower_dsb.log)"
unlock; lock
EXTRA_ENV="$BEST DRAMA3_TOWER_DS=1" $D/run_gen.sh $P/c_ds.log --interactive --lines all $R --out $P/c_ds; echo "c_ds $(tail -1 $P/c_ds.log) $(date +%T)"
unlock; lock
EXTRA_ENV="$BEST DRAMA3_TOWER_DS=1" $D/run_gen.sh $P/c_dsb.log --interactive --lines all $R --out $P/c_dsb --tower-mlp bfp8; echo "c_dsb $(tail -1 $P/c_dsb.log) $(date +%T)"
unlock
python3 $P/rtf.py $P/c_ds $P/c_dsb
echo "=== chain10 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN10_DONE
