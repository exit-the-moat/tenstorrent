#!/bin/bash
# Day 6: re-render the narration lines with Whisper match < 0.85 with seeds 1-3, keep the best take
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
ids=$1
out=/opt/ep0/day6/retry; nar=/opt/ep0/day6/narration
mkdir -p $out
echo "=== retry $ids start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
rm -f $out/power.stop
python3 $D/power_log.py log $out/power.jsonl $out/power.stop &
$D/run_gen.sh $out/gen.log --fast --lines $ids --lines-file /opt/ep0/voice/card_lines.json --seeds 1,2,3 --ref /opt/ep0/voice/ref_v2.wav --out $out
touch $out/power.stop; wait
python3 $D/power_log.py merge $out/power.jsonl $out/timing.json
/opt/ep0/run.sh "DRAMA3_LINES_FILE=/opt/ep0/voice/card_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day6_pick $nar $out" > $out/pick.log 2>&1
(cd $nar && sha256sum *.wav *.frames.pt timing.json sanity.json picks.json > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK day6")
echo "=== retry end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo RETRY_DONE
