#!/bin/bash
# port10: defaults bit-identical? --fast vs day4b/out_v2, --interactive vs port6/i_qdt (card $CARD, default 1); name suffix $1
source /opt/ep0/port10/env10.sh
C=${CARD:-1}; s=${1:-}
lockc $C
EXTRA_ENV="TT_VISIBLE_DEVICES=$C" $D/run_gen.sh $P/r_def$s.log --fast --lines all $R --out $P/r_def$s
EXTRA_ENV="TT_VISIBLE_DEVICES=$C" $D/run_gen.sh $P/i_base$s.log --interactive --lines all $R --out $P/i_base$s
unlockc $C
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/r_def$s /opt/ep0/day4b/out_v2" 2>&1 | grep "^[a-z_]* {" > $P/r_def$s.cmp; echo "r_def$s $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/r_def$s.cmp) of 8 identical to day4b/out_v2"
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $P/i_base$s /opt/ep0/port6/i_qdt" 2>&1 | grep "^[a-z_]* {" > $P/i_base$s.cmp; echo "i_base$s $(grep -c "\"frames_identical\": true, \"wav_identical\": true" $P/i_base$s.cmp) of 8 identical to i_qdt"
echo "$(iochk)"; echo DEFS_DONE
