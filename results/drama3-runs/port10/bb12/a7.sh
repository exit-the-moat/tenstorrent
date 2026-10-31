source /opt/ep0/port10/bb12/env.sh
chk cbp3 "DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160" tests.check_bb_prefetch2
grep "^ab" $Q/cbp3.log
echo A7_DONE
