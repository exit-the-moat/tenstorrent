#!/bin/bash
cd /opt/ep0/port10
while pgrep -f chainD.sh >/dev/null; do sleep 15; done
CARD=1 ./run10.sh i_dd8 8 short "DRAMA3_MOE_BATCH_DEDUP=1"
CARD=1 ./run10.sh i_vd8 8 short "DRAMA3_CB_VDRAW=1"
for d in i_dd8 i_vd8; do echo "$d vs base: $(/opt/ep0/run.sh "python /opt/ep0/port10/cmpf.py /opt/ep0/port10/$d /opt/ep0/port10/b_short_8_d0" 2>&1 | tail -1)"; done
echo CHAIN_E_DONE
