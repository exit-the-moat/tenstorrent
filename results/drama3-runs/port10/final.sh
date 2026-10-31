#!/bin/bash
# port10 counted runs: card 0 alone (card 1 idle), one at a time. C4 = all port10 levers.
cd /opt/ep0/port10
source /opt/ep0/port10/env10.sh
echo "start $(date +%T) $(iochk)"
while [ -e /opt/ep0/card1.lock ]; do sleep 10; done
C4="DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1 DRAMA3_KV_COMPACT=1 DRAMA3_STREAM_ENDS=10 DRAMA3_CODEC_DEV_TRACE=1 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1 QWEN36_GROUPED_COMPACT=1 DRAMA3_CODEC_WINDOWS=32,96"
./run10.sh fb_short_8 8 short ""
for B in 8 12 16 24; do ./run10.sh f_short_$B $B short "$C4"; done
./run10.sh fb_long_8 8 long ""
for B in 8 16 24; do ./run10.sh f_long_$B $B long "$C4"; done
/opt/ep0/run.sh "TT_VISIBLE_DEVICES=0 python /opt/ep0/smoke.py" > smoke_final.log 2>&1; grep SMOKE smoke_final.log
echo "end $(date +%T) $(iochk)"
echo FINAL_DONE
