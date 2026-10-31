#!/bin/bash
# cc13 step 1: one-CQ and two-CQ runs of the synthetic sub-device overlap test, card 1, one lock
source /opt/ep0/port10/env10.sh
lockc 1
for q in 2; do
  EXTRA_ENV="TT_VISIBLE_DEVICES=1" timeout 300 $D/run_check.sh /opt/ep0/port10/cc13/conc_q$q.log tests.cc13_conc --cqs $q --out /opt/ep0/port10/cc13/conc_q$q.json
done
unlockc 1
echo "conc $(tail -n1 /opt/ep0/port10/cc13/conc_q1.log) $(tail -n1 /opt/ep0/port10/cc13/conc_q2.log) $(iochk)" > /opt/ep0/port10/cc13/conc.status
echo R1_DONE >> /opt/ep0/port10/cc13/conc.status
