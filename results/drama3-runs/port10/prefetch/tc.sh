#!/bin/bash
# tower check flag off / on (card 1): tc.sh <tag> <on|off|both>
source /opt/ep0/port10/prefetch/ienv.sh
cd /opt/ep0/port10
for v in off on; do
  [ "$2" != both ] && [ "$2" != $v ] && continue
  F=""; [ $v = on ] && F="DRAMA3_TOWER_PREFETCH=1 ${PFENV:-}"
  CARD=1 TMO=600 ./t1.sh prefetch/tc_$1_$v "$IENV $F" tests.check_tower_device --mlp bfp8 --live-rope 1 --out /opt/ep0/port10/prefetch/tc_$1_$v.json
done
echo TC_DONE
