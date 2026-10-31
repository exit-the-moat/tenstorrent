source /opt/ep0/port10/bb12/env.sh
chk golden_on160 "DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160" tests.check_paged_backbone --out $Q/golden_on160.json
echo A2_DONE
