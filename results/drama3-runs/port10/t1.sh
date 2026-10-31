#!/bin/bash
# port10: run one test module on card $CARD (default 1) under its lock: t1.sh <name> "<extra env>" <module> <args...>
source /opt/ep0/port10/env10.sh
C=${CARD:-1}; n=$1; shift; ex=$1; shift
lockc $C
EXTRA_ENV="TT_VISIBLE_DEVICES=$C $B7 $ex" timeout ${TMO:-1800} $D/run_check.sh $P/$n.log "$@"
unlockc $C
echo "$n $(tail -n 1 $P/$n.log) $(iochk)"
echo T1_DONE
