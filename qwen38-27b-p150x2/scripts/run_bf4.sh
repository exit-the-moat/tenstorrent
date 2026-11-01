#!/bin/bash
# Naive BF4 on 2 cards (goal item "if time allows"): every linear weight BF4 (patch_bf4.py), else as the default run.
L=/opt/ep1/serve/ep1-vllm-qwen38-p150x2-naivebf4.log
until grep -q "Application startup complete" $L 2>/dev/null; do
  grep -q -E "EngineCore failed|TT_THROW|TT_FATAL" $L 2>/dev/null && { echo "SERVER FAILED"; echo "ALL DONE"; exit 1; }
  sleep 10
done
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a+ep1-gdn-v5+naivebf4 EP1_POWER_CARDS=0,1
echo "PHASE bf4ladder START $(date -Is)"
python3 /opt/ep1/bench/bench.py --config 2card-2cable-naivebf4 --cards 2 --cables 2 --precision naive-bf4 \
  --users 1,4,8 --inputs 1024,4096 --repeats 2 && echo "PHASE bf4ladder DONE $(date -Is)" || echo "PHASE bf4ladder FAILED"
echo "PHASE bf4gsm START $(date -Is)"
python3 /opt/ep1/eval/gsm8k.py run --config 2card-2cable-naivebf4 --precision naive-bf4 --workers 8 \
  && echo "PHASE bf4gsm DONE $(date -Is)" || echo "PHASE bf4gsm FAILED"
echo "ALL DONE $(date -Is)"
