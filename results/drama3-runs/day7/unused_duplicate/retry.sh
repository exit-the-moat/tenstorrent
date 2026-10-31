#!/bin/bash
# Day 7: narration retries (s04-05, s05-02, the moat lines incl. the "mote" spelling) with seeds 1-3, default
# --fast pipeline, sequential, ref_v2; power sampled; Whisper small + base scoring (tests/day7_retry_pick.py);
# sync + drop caches + SHA256SUMS.  Output: /opt/ep0/day6/narration_retry/<id>_<variant>_s<seed>.wav
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
out=/opt/ep0/day6/narration_retry; L=/opt/ep0/day7/retry_lines.json
mkdir -p $out
echo "=== day7 retry start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
ids=$(python3 -c "import json;print(','.join(l['id'] for l in json.load(open('$L'))))")
rm -f $out/power.stop
python3 $D/power_log.py log $out/power.jsonl $out/power.stop &
$D/run_gen.sh $out/gen.log --fast --lines $ids --lines-file $L --seeds 1,2,3 --ref /opt/ep0/voice/ref_v2.wav --out $out
touch $out/power.stop; wait
python3 $D/power_log.py merge $out/power.jsonl $out/timing.json
echo "=== gen $(tail -1 $out/gen.log)"
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day7_retry_pick $L /opt/ep0/day6/narration $out" > $out/pick.log 2>&1
echo "=== pick $(tail -1 $out/pick.log)"
(cd $out && sha256sum *.wav *.frames.pt timing.json picks.json > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK day7 retry")
echo "=== day7 retry end $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
echo DAY7_RETRY_DONE
