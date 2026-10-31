#!/bin/bash
# port10: check_batch_numerics on card $CARD (default 1) for sizes $1, out name $2, extra env $3
source /opt/ep0/port10/env10.sh
C=${CARD:-1}
lockc $C
EXTRA_ENV="TT_VISIBLE_DEVICES=$C $B7 $3" timeout 1800 $D/run_check.sh $P/$2.log tests.check_batch_numerics --sizes $1 --out $P/$2.json
unlockc $C
echo "$2 $(tail -n 1 $P/$2.log) $(iochk)"
echo CBN_DONE
