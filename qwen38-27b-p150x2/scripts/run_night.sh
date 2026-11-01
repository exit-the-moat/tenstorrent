#!/bin/bash
# Episode 1 overnight measurement sequence (2026-11-01). Assumes ep1-vllm (P150x2, default) is already up on :8085 and
# /opt/ep0/card.lock is held. Each phase prints PHASE <name> START / DONE / FAILED; the last line is ALL DONE.
# Usage: run_night.sh [first_phase]   phases: ladder2 prefill2 gsm2
set -u
cd /opt/ep1
export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a+ep1-gdn-v5 EP1_POWER_CARDS=0,1
B="python3 /opt/ep1/bench/bench.py"
START=${1:-ladder2}
go=0

wait_ready() {  # $1 log file; returns 0 when vLLM is up, 1 on failure
  for _ in $(seq 1 540); do  # 90 min max
    grep -q "Application startup complete" "$1" 2>/dev/null && return 0
    grep -q -E "Traceback|TT_THROW|TT_FATAL" "$1" 2>/dev/null && return 1
    [ "$(podman inspect -f '{{.State.Running}}' ep1-vllm 2>/dev/null)" = "true" ] || return 1
    sleep 10
  done
  return 1
}
phase() { [ "$1" = "$START" ] && go=1; [ $go = 1 ]; }
io() { echo "dmesg I/O errors: $(dmesg --level=err | grep -c 'I/O error')"; }

io
if phase ladder2; then
  echo "PHASE ladder2 START $(date -Is)"
  $B --config 2card-2cable-default --cards 2 --cables 2 --precision default-bfp8-bfp4mlp \
     --users 1,4,8 --inputs 1024,4096,32768 --repeats 2 && echo "PHASE ladder2 DONE $(date -Is)" || echo "PHASE ladder2 FAILED"
fi
if phase prefill2; then
  echo "PHASE prefill2 START $(date -Is)"
  $B --config 2card-2cable-default --cards 2 --cables 2 --precision default-bfp8-bfp4mlp \
     --users 1 --inputs 8192 --repeats 2 && echo "PHASE prefill2 DONE $(date -Is)" || echo "PHASE prefill2 FAILED"
fi
if phase gsm2; then
  echo "PHASE gsm2 START $(date -Is)"
  python3 /opt/ep1/eval/gsm8k.py run --config 2card-2cable-default --precision default-bfp8-bfp4mlp --workers 8 \
    && echo "PHASE gsm2 DONE $(date -Is)" || echo "PHASE gsm2 FAILED"
fi
# warm2 / one: dropped 05:30 (1-card done separately; a restart needs the patch flags)
io
echo "ALL DONE $(date -Is)"
