#!/bin/bash
# Episode 1 fallback (2026-11-01, 03:20): TP=2 didn't start within its time box, so one p150a (card 0, MESH=P150,
# unpatched stock code), card 1 idle. Starts ep1-vllm, waits, runs the ladders and GSM8K. Last line ALL DONE.
# 03:30: first 1-card start ran out of DRAM allocating a 68,800-token KV pool (weights 24.6 GB of ~33 GB): pool 24,576, 16k context.
# 03:45: start 2 failed: single-device GDN state is batch 1 only ("recurrent_state shape mismatch [4,...] != [1,...]"): SEQS=1.
set -u
cd /opt/ep1
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a
B="python3 /opt/ep1/bench/bench.py --config 1card-default --system 1x-p150a --cards 1 --cables 0 --precision default-bfp8-bfp4mlp"
L=/opt/ep1/serve/ep1-vllm-qwen38-p150-default.log
echo "dmesg I/O errors: $(dmesg --level=err | grep -c 'I/O error')"
podman rm -f ep1-vllm >/dev/null 2>&1
MESH=P150 DEVS=0 LEN=16384 SEQS=1 POOL=24576 /opt/ep1/serve/start-ep1.sh
echo "PHASE start1 START $(date -Is)"
ok=0
for _ in $(seq 1 540); do
  grep -q "Application startup complete" $L 2>/dev/null && { ok=1; break; }
  grep -q -E "EngineCore failed|TT_THROW|TT_FATAL" $L 2>/dev/null && break
  [ "$(podman inspect -f '{{.State.Running}}' ep1-vllm 2>/dev/null)" = "true" ] || break
  sleep 10
done
[ $ok = 1 ] || { echo "PHASE start1 FAILED $(date -Is)"; echo "ALL DONE"; exit 1; }
echo "PHASE start1 DONE $(date -Is)"
echo "PHASE ladder1 START $(date -Is)"
$B --users 1 --inputs 1024,4096 --repeats 2 && echo "PHASE ladder1 DONE $(date -Is)" || echo "PHASE ladder1 FAILED"
echo "PHASE long1 START $(date -Is)"
$B --users 1 --inputs 8192 --repeats 2 && echo "PHASE long1 DONE $(date -Is)" || echo "PHASE long1 FAILED"
echo "PHASE gsm1 START $(date -Is)"
python3 /opt/ep1/eval/gsm8k.py run --config 1card-default --precision default-bfp8-bfp4mlp --workers 1 \
  && echo "PHASE gsm1 DONE $(date -Is)" || echo "PHASE gsm1 FAILED"
echo "dmesg I/O errors: $(dmesg --level=err | grep -c 'I/O error')"
echo "ALL DONE $(date -Is)"
