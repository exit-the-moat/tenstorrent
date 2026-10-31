#!/bin/bash
cd /opt/ep0/port10
source /opt/ep0/port10/prefetch/ienv.sh
PF='DRAMA3_TOWER_PREFETCH_GCB_KB=128 DRAMA3_TOWER_PREFETCH_PAGE_KB=64'
PFENV="$PF" prefetch/tc.sh e on
for v in off on; do
  F=''; [ $v = on ] && F="DRAMA3_TOWER_PREFETCH=1 $PF"
  CARD=1 TMO=600 ./t1.sh prefetch/pt_$v "$IENV $F" tests.profile_frame7 --mode time --tower-mlp bfp8 --out /opt/ep0/port10/prefetch/pt_$v.json
done
echo CHAIN_DONE
