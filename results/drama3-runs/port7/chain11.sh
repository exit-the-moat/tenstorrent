#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
BEST="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1"
echo "=== chain11 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
EXTRA_ENV="$I7 QWEN36_GROUPED_COMPACT=1" $D/run_check.sh $P/prefix_compact.log tests.check_prefix_cache --out $P/prefix_compact.json; echo "prefix_compact $(tail -1 $P/prefix_compact.log)"
unlock; lock
EXTRA_ENV="$I7" $D/run_check.sh $P/prefix_base.log tests.check_prefix_cache --out $P/prefix_base.json; echo "prefix_base $(tail -1 $P/prefix_base.log)"
unlock; lock
EXTRA_ENV="$BEST QWEN36_GROUPED_COMPACT=1" $D/run_gen.sh $P/c_cmp.log --interactive --lines all $R --out $P/c_cmp; echo "c_cmp $(tail -1 $P/c_cmp.log) $(date +%T)"
unlock
python3 $P/rtf.py $P/r5 $P/c_cmp
echo CHAIN11_DONE
