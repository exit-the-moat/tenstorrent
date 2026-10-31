#!/bin/bash
# port10: single stream (--interactive, 8 lines) with and without the codec levers, card 0 alone, 2 repeats each
source /opt/ep0/port10/env10.sh
CX="DRAMA3_CODEC_DEV_TRACE=1 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1"
lockc 0
for r in 1 2; do
  for n in base cx; do
    o=$P/solo_${n}_$r; [ $n = cx ] && ex="$CX" || ex=""
    pw_start $o 0
    EXTRA_ENV="TT_VISIBLE_DEVICES=0 $ex" $D/run_gen.sh $o.log --interactive --lines all $R --out $o
    pw_stop $o
    echo "solo_${n}_$r $(tail -n1 $o.log)"
  done
done
unlockc 0
for r in 1 2; do /opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/solo_cx_$r $P/solo_base_$r" 2>&1 | grep -c "\"frames_identical\": true, \"wav_identical\": true" | sed "s/^/identical lines r$r: /"; done
echo "$(iochk)"; echo SOLO_DONE
