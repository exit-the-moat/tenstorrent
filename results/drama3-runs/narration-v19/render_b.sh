#!/bin/bash
# Episode 0 Day 6: the episode narration (card_lines.json, text = spoken) with ref_v2 on the fast pipeline (one
# line after the other), card power sampled alongside, Whisper vs the written form (sanity.json), sync + drop
# caches, SHA256SUMS.   render_day6.sh [out dir] [extra generator args...]   (EXTRA_ENV as for run_gen.sh)
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
set -u
out=${1:-/opt/ep0/day6/narration}; shift || true
mkdir -p "$out"
echo "=== v19b start $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error")"
rm -f "$out/power.stop"
python3 $D/power_log.py log "$out/power.jsonl" "$out/power.stop" &
$D/run_gen.sh "$out/gen.log" --fast --lines all --lines-file /opt/ep0/voice/v19b_lines.json --ref /opt/ep0/voice/ref_v2.wav --out "$out" "$@"
touch "$out/power.stop"; wait
python3 $D/power_log.py merge "$out/power.jsonl" "$out/timing.json"
/opt/ep0/run.sh "DRAMA3_LINES_FILE=/opt/ep0/voice/v19b_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_day4 $out" > "$out/summary.log" 2>&1
# sums of the bytes as written (page cache), then re-read from disk after sync + drop caches
(cd "$out" && sha256sum *.wav *.frames.pt timing.json sanity.json > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK day6")
echo "=== v19b end $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error") $(tail -1 $out/gen.log)"
echo DAY6_RENDER_DONE
