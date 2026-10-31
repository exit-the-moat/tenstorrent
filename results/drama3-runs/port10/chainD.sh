#!/bin/bash
cd /opt/ep0/port10
while pgrep -f "t1.sh cbn_24" >/dev/null; do sleep 15; done
C1="DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1 DRAMA3_KV_COMPACT=1"
CARD=1 ./run10.sh x24 24 short "$C1"
echo CHAIN_D_DONE
