#!/bin/bash
# Episode 0 Day 4 exit runs: the 8 lines with the old reference (ref_long) and with ref_v2 on the fast pipeline,
# with card power sampled alongside; then power merged into each timing.json, sync + drop caches, SHA256SUMS.
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
set -u
for spec in "ref_long:/opt/ep0/day4/out" "ref_v2:/opt/ep0/day4/out_v2"; do
  ref=${spec%%:*}; out=${spec#*:}
  mkdir -p "$out"
  echo "=== $ref start $(date +%T) io_errors=$(dmesg --level=err | grep -c 'I/O error')"
  rm -f "$out/power.stop"
  python3 $D/power_log.py log "$out/power.jsonl" "$out/power.stop" &
  $D/run_gen.sh "$out/gen.log" --fast --lines all --ref /opt/ep0/voice/$ref.wav --out "$out"
  touch "$out/power.stop"; wait
  python3 $D/power_log.py merge "$out/power.jsonl" "$out/timing.json"
  sync; echo 3 > /proc/sys/vm/drop_caches
  (cd "$out" && sha256sum *.wav *.frames.pt timing.json > SHA256SUMS && sha256sum -c --quiet SHA256SUMS && echo "SUMS_OK $ref")
  echo "=== $ref end $(date +%T) io_errors=$(dmesg --level=err | grep -c 'I/O error') $(tail -1 $out/gen.log)"
done
echo ALL_RENDERS_DONE
