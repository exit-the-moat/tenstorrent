#!/bin/bash
# Runs the 1-card ladders after run_night1.sh (GSM8K) is done; the bench failed there on HTTP 400 (stream_options on a
# non-streaming call, fixed 04:10). Server ep1-vllm (P150, 1 sequence) stays up. Last line ALL DONE.
until grep -q "ALL DONE" /opt/ep1/night1.log; do sleep 15; done
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a
B="python3 /opt/ep1/bench/bench.py --config 1card-default --system 1x-p150a --cards 1 --cables 0 --precision default-bfp8-bfp4mlp"
echo "PHASE ladder1b START $(date -Is)"
$B --users 1 --inputs 1024,4096,8192 --repeats 2 && echo "PHASE ladder1b DONE $(date -Is)" || echo "PHASE ladder1b FAILED"
echo "dmesg I/O errors: $(dmesg --level=err | grep -c "I/O error")"
echo "ALL DONE $(date -Is)"
