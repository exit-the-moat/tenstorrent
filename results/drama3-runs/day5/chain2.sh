#!/bin/bash
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
echo "=== chain2 start $(date +%T)"
EXTRA_ENV="QWEN36_EXPERT_GU_BFP8_LAYERS=0,27,28,30,32,33,36,37" $D/render_day4b.sh /opt/ep0/day5/out_bfp8_8
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.summarize_day4 /opt/ep0/day5/out_bfp8_8" > /opt/ep0/day5/out_bfp8_8/summary.log 2>&1
(cd /opt/ep0/day5/out_bfp8_8 && sha256sum sanity.json >> SHA256SUMS)
$D/render_day6.sh /opt/ep0/day6/narration
echo "=== chain2 end $(date +%T)"
echo CHAIN2_DONE
