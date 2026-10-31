source /opt/ep0/port10/bb12/env.sh
chk bits_bb "" tests.check_prefetch_bits_bb
grep "^bits" $Q/bits_bb.log
chk golden_off "" tests.check_paged_backbone --out $Q/golden_off.json
echo A3_DONE
