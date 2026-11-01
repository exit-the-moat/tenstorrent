#!/bin/bash
# 2-card ladder at 128-token prompts (the goal list; decode only: 128 tokens is too short for a prefill number, so the
# prefill_tps of these rows is not used). Waits for the server, runs, prints ALL DONE.
L=/opt/ep1/serve/ep1-vllm-qwen38-p150x2-default.log
until grep -q "Application startup complete" $L; do
  grep -q -E "EngineCore failed|TT_THROW|TT_FATAL" $L && { echo "SERVER FAILED"; echo "ALL DONE"; exit 1; }
  sleep 10
done
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a+ep1-gdn-v5 EP1_POWER_CARDS=0,1
echo "PHASE l128 START $(date -Is)"
python3 /opt/ep1/bench/bench.py --config 2card-2cable-default --cards 2 --cables 2 --precision default-bfp8-bfp4mlp \
  --users 1,4,8 --inputs 128 --repeats 2 && echo "PHASE l128 DONE $(date -Is)" || echo "PHASE l128 FAILED"
echo "ALL DONE $(date -Is)"
