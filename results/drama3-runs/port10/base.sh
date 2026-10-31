#!/bin/bash
# port10 baseline on card 0: q_short_8 and q_long_8 (32 jobs at t=0, greedy), B7 flags, as port9 chain_q.sh
source /opt/ep0/port10/env10.sh
C=${CARD:-0}
echo "start $(date +%T) $(iochk) card=$C"
lockc $C
for W in short long; do
  o=$P/b_${W}_8_d$C; mkdir -p $o; pw_start $o $C
  if [ $W = short ]; then A="--lines all"; T=1200; else A="--lines-file /opt/ep0/port9/long_lines.json --lines all --max-frames 1200"; T=2400; fi
  EXTRA_ENV="TT_VISIBLE_DEVICES=$C $B7" timeout $T $D/run_check.sh $o/gen.log host.cbatch --batch-size 8 $A --repeat 4 --arrivals 0 --policy greedy $R --out $o
  pw_stop $o; echo "b_${W}_8 $(tail -n 1 $o/gen.log) $(tail -c 300 $o/power.txt | tr "\n" " ") $(date +%T)"
done
unlockc $C
echo "$(iochk)"
echo BASE_DONE
