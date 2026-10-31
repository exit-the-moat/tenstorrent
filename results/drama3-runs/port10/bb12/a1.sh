source /opt/ep0/port10/bb12/env.sh
chk golden_on "DRAMA3_BB_PREFETCH=1" tests.check_paged_backbone --out $Q/golden_on.json
echo A1_DONE
