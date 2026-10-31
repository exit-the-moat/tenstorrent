#!/bin/bash
cd /opt/ep0/port10
while pgrep -f chainF.sh >/dev/null; do sleep 15; done
C2="DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1 DRAMA3_KV_COMPACT=1 DRAMA3_STREAM_ENDS=10"
./run10.sh g16_t4 16 short "$C2 DRAMA3_CB_THREADS=4"
./run10.sh g16_gc 16 short "$C2 QWEN36_GROUPED_COMPACT=1"
echo CHAIN_G_DONE
