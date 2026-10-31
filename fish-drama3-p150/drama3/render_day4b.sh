#!/bin/bash
# Episode 0 Day 4b exit run: the 8 lines with ref_v2 on the Day 4b pipeline (voice prefix cache, device-side
# codebook sampling), card power sampled alongside; power merged into timing.json, sync + drop caches, SHA256SUMS.
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
set -u
out=${1:-/opt/ep0/day4b/out_v2}
mkdir -p "$out"
echo "=== ref_v2 start $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error")"
rm -f "$out/power.stop"
python3 $D/power_log.py log "$out/power.jsonl" "$out/power.stop" &
$D/run_gen.sh "$out/gen.log" --fast --lines all --ref /opt/ep0/voice/ref_v2.wav --out "$out"
touch "$out/power.stop"; wait
python3 $D/power_log.py merge "$out/power.jsonl" "$out/timing.json"
sync; echo 3 > /proc/sys/vm/drop_caches
(cd "$out" && sha256sum *.wav *.frames.pt timing.json > SHA256SUMS && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK ref_v2")
echo "=== ref_v2 end $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error") $(tail -1 $out/gen.log)"
echo ALL_RENDERS_DONE
