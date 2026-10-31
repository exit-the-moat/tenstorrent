#!/bin/bash
# port8 item 1: fused add + norm: frame times (port7 flags vs + ADDNORM), tower check, golden backbone check
source /opt/ep0/port8/env8.sh
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
lock
EXTRA_ENV="$I7 $P7F" $D/run_check.sh $P/ft_base.log tests.profile_frame7 --mode time --out $P/ft_base.json; echo "ft_base $(tail -1 $P/ft_base.log) $(date +%T)"
EXTRA_ENV="$I7 $P7F DRAMA3_ADDNORM_FUSED=1" $D/run_check.sh $P/ft_an.log tests.profile_frame7 --mode time --out $P/ft_an.json; echo "ft_an $(tail -1 $P/ft_an.log) $(date +%T)"
EXTRA_ENV="$I7 DRAMA3_TOWER_ETABLE=1 DRAMA3_ADDNORM_FUSED=1" $D/run_check.sh $P/tower_an.log tests.check_tower_device; echo "tower_an $(tail -1 $P/tower_an.log) $(date +%T)"
EXTRA_ENV="$I7 DRAMA3_ADDNORM_FUSED=1" $D/run_check.sh $P/golden_an.log tests.check_paged_backbone --out $P/golden_an.json; echo "golden_an $(tail -1 $P/golden_an.log) $(date +%T)"
unlock
echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1)"
echo CHAIN1_DONE
