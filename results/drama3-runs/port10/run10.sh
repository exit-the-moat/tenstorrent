#!/bin/bash
# port10: one full-queue cbatch run on card $CARD (default 0): run10.sh <name> <B> <short|long> "<extra env>"
source /opt/ep0/port10/env10.sh
C=${CARD:-0}; n=$1; B=$2; W=$3; ex=$4
o=$P/$n; mkdir -p $o
if [ $W = short ]; then A="--lines all"; T=1800; else A="--lines-file /opt/ep0/port9/long_lines.json --lines all --max-frames 1200"; T=3000; fi
lockc $C
pw_start $o $C
EXTRA_ENV="TT_VISIBLE_DEVICES=$C $B7 $ex" timeout $T $D/run_check.sh $o/gen.log host.cbatch --batch-size $B $A --repeat 4 --arrivals 0 --policy greedy $R --out $o
pw_stop $o
unlockc $C
echo "$n $(tail -n 1 $o/gen.log) $(date +%T) $(iochk)"
grep -q "EXIT 0" $o/gen.log || grep -E "Error|assert|TT_FATAL|TT_THROW" $o/gen.log | tail -n 3 | cut -c1-300
[ -f $o/timing.json ] && python3 /opt/ep0/port9/sum_st.py $o
