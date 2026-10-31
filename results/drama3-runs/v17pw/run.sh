#!/bin/bash
# v17 scoreboard: the live preset (run_gen.sh --interactive) on the 8 test lines, ref_v2, with card power sampled.
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3; o=/opt/ep0/v17pw/out; mkdir -p $o
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
rm -f $o/power.stop; python3 $D/power_log.py log $o/power.jsonl $o/power.stop &
$D/run_gen.sh $o/gen.log --interactive --lines all --ref /opt/ep0/voice/ref_v2.wav --out $o
touch $o/power.stop; sleep 3
python3 /opt/ep0/port6/power_sum.py $o/power.jsonl $o/timing.json > $o/power.txt 2>&1
echo "gen $(tail -1 $o/gen.log)"; grep -h "DRAM" $o/gen.log; cat $o/power.txt
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
echo DONE
