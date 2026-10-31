source /opt/ep0/port10/bb12/env.sh
chk cbp2 "DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160" tests.check_bb_prefetch2
grep "^state_diff|^addnorm"\|^addnorm" $Q/cbp2.log
echo A6_DONE
