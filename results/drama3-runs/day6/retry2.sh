#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
out=/opt/ep0/day6/narration_retry
echo "=== retry2 start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
rm -f $out/power.stop; python3 $D/power_log.py log $out/power.jsonl $out/power.stop &
$D/run_gen.sh $out/gen.log --fast --lines all --lines-file $out/variants.json --seeds 1,2,3 --ref /opt/ep0/voice/ref_v2.wav --out $out
touch $out/power.stop; wait
python3 $D/power_log.py merge $out/power.jsonl $out/timing.json > /dev/null
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day6_variants $out $out/variants.json" > $out/score.log 2>&1
(cd $out && sha256sum *.wav *.frames.pt timing.json picks.json variants.json > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo SUMS_OK retry2)
echo "=== retry2 end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error") $(tail -1 $out/gen.log)"
echo RETRY2_DONE
