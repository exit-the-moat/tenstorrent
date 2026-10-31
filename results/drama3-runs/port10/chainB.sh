#!/bin/bash
cd /opt/ep0/port10
while pgrep -f chainA.sh >/dev/null; do sleep 15; done
C1="DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1"
./run10.sh c1_s16 16 short "$C1"
./run10.sh c1e_s16 16 short "$C1 DRAMA3_STREAM_ENDS=10"
./run10.sh c1_s8 8 short "$C1"
echo CHAIN_B_DONE
