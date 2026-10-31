#!/bin/bash
# port7: the 8 test lines (ref_v2) with run_gen.sh --interactive + EXTRA_ENV, card power alongside, merged into
# timing.json, sync + drop caches, SHA256SUMS.   render7.sh <out dir> [generator args...]
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
set -u
out=$1; shift
mkdir -p "$out"
echo "=== $out start $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error")"
rm -f "$out/power.stop"
python3 $D/power_log.py log "$out/power.jsonl" "$out/power.stop" &
$D/run_gen.sh "$out/gen.log" --interactive --lines all --ref /opt/ep0/voice/ref_v2.wav --out "$out" "$@"
touch "$out/power.stop"; wait
python3 $D/power_log.py merge "$out/power.jsonl" "$out/timing.json"
sync; echo 3 > /proc/sys/vm/drop_caches
(cd "$out" && sha256sum *.wav *.frames.pt timing.json > SHA256SUMS && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK $out")
echo "=== $out end $(date +%T) io_errors=$(dmesg --level=err | grep -c "I/O error") $(tail -1 $out/gen.log)"
