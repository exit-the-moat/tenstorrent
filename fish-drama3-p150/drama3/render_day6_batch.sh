#!/bin/bash
# Episode 0 Day 6 in batch mode (Day 5b): the 38 narration lines, B at a time (batches of similar length), ref_v2,
# power alongside, Whisper vs the written form + throughput summary, SHA256SUMS (as written, then from disk).
#   render_day6_batch.sh <B> [out dir]
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
set -u
B=$1; out=${2:-/opt/ep0/day6/narration_b$B}
mkdir -p "$out"
echo "=== day6 batch $B start $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error")"
rm -f "$out/power.stop"
python3 $D/power_log.py log "$out/power.jsonl" "$out/power.stop" &
$D/run_gen.sh "$out/gen.log" --batch --batch-size $B --lines all --lines-file /opt/ep0/voice/card_lines.json --ref /opt/ep0/voice/ref_v2.wav --out "$out"
touch "$out/power.stop"; wait
python3 $D/power_log.py merge "$out/power.jsonl" "$out/timing.json" > /dev/null
/opt/ep0/run.sh "DRAMA3_LINES_FILE=/opt/ep0/voice/card_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_batch $out" > "$out/summary.log" 2>&1
(cd "$out" && sha256sum *.wav *.frames.pt timing.json sanity.json batch_summary.json > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK day6 batch $B")
echo "=== day6 batch $B end $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error") $(tail -1 $out/gen.log)"
echo DAY6_BATCH_DONE
