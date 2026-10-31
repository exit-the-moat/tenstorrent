#!/bin/bash
# One voice: the 8 test lines with --interactive (live use) and --fast (episode renders), then the backbone
# golden check with the --interactive levers. Writes <out>/{interactive,fast}/timing.json (TTFA, RTF, playback check)
# and <out>/golden.json (only if the FP32 reference for your voice exists, see the README). Usage: bench/solo.sh [out dir, default /opt/ep0/bench]
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/env.sh
P=${1:-/opt/ep0/bench}; mkdir -p $P
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
for m in interactive fast; do
  $D/run_gen.sh $P/$m.log --$m --lines all --ref $REF --out $P/$m
  echo "$m $(tail -1 $P/$m.log) $(date +%T)"
done
[ -d /opt/ep0/day3/golden_voice ] && EXTRA_ENV="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1" $D/run_check.sh $P/golden.log tests.check_paged_backbone --out $P/golden.json
[ -f $P/golden.log ] && echo "golden $(tail -1 $P/golden.log) $(date +%T)"
echo DONE
