#!/bin/bash
# Power pass for the 1-card config: the ladder rows (bench1.log) summed tt-smi board power over BOTH cards, incl. the idle
# card 1. Same request shape (1 user, 4k prompt, warm-up + 2 repeats), power from card 0 only. Config 1card-default-power.
until grep -q "ALL DONE" /opt/ep1/bench1.log; do sleep 10; done
mv /opt/ep1/bench/bench.py.new /opt/ep1/bench/bench.py
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a EP1_POWER_CARDS=0
echo "PHASE power1 START $(date -Is)"
python3 /opt/ep1/bench/bench.py --config 1card-default-power --system 1x-p150a --cards 1 --cables 0 \
  --precision default-bfp8-bfp4mlp --users 1 --inputs 4096 --repeats 2 && echo "PHASE power1 DONE $(date -Is)" || echo "PHASE power1 FAILED"
echo "ALL DONE $(date -Is)"
