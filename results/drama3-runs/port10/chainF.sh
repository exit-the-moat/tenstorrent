#!/bin/bash
# port10: exploratory runs, card 0 only (one at a time: host CPU contention distorts host ms)
cd /opt/ep0/port10
while pgrep -f chainC.sh >/dev/null; do sleep 15; done
C1="DRAMA3_MOE_BATCH_DEDUP=1 DRAMA3_MOE_DEDUP_MASK=1 DRAMA3_CB_VDRAW=1 DRAMA3_KV_COMPACT=1"
./run10.sh x24 24 short "$C1"
./run10.sh i_dd8 8 short "DRAMA3_MOE_BATCH_DEDUP=1"
./run10.sh i_vd8 8 short "DRAMA3_CB_VDRAW=1"
for d in i_dd8 i_vd8; do echo "$d vs base: $(/opt/ep0/run.sh "python /opt/ep0/port10/cmpf.py /opt/ep0/port10/$d /opt/ep0/port10/b_short_8_d0" 2>&1 | tail -1)"; done
./run10.sh x16_t4 16 short "$C1 DRAMA3_CB_THREADS=4"
echo CHAIN_F_DONE
