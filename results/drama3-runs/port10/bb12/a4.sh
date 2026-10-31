source /opt/ep0/port10/bb12/env.sh
chk cbp "DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160" tests.check_bb_prefetch
grep "copies\|mismatching" $Q/cbp.log
echo A4_DONE
