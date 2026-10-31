#!/bin/bash
# Day 5: CPU FP32 sims of the voice golden with the device weight formats (BFP4 vs BFP8 expert gate/up)
for q in qwen36 bfp8; do
  echo "=== sim $q start $(date +%T)"
  /opt/ep0/run.sh "python -m models.demos.blackhole.drama3.reference.cpu_layer_ref --rows /opt/ep0/day3/close.rows_teacher24.pt --quant $q --threads 14 --out /opt/ep0/day5/sim_$q"
  echo "=== sim $q rc=$? end $(date +%T)"
done
echo SIMS_DONE
