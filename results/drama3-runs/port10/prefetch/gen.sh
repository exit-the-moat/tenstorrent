#!/bin/bash
# 8 interactive lines on card 1: gen.sh <name> "<extra env>" -> /opt/ep0/port10/prefetch/<name>/ + <name>.log, then cmp vs i_bf8
source /opt/ep0/port10/env10.sh
cd /opt/ep0/tt-metal/models/demos/blackhole/drama3
O=/opt/ep0/port10/prefetch/$1
lockc 1
EXTRA_ENV="TT_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1 $2" /opt/ep0/port10/prefetch/pfrun.sh "--out $O" ${TMO:-900} ./run_gen.sh $O.log --interactive --lines all --ref /opt/ep0/voice/ref_v2.wav --out $O
unlockc 1
tail -1 $O.log
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $O /opt/ep0/port10/i_bf8" > $O.cmp 2>&1
grep -E 'identical|frames|wav' $O.cmp | tail -12
echo "$(iochk)"
echo GEN_DONE
