#!/bin/bash
# port8: frame times of the candidate stacks (profile_frame7 --mode time)
source /opt/ep0/port8/env8.sh
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
f() { n=$1; shift; EXTRA_ENV="$I7 $P7F $*" timeout 480 $D/run_check.sh $P/ft_$n.log tests.profile_frame7 --mode time --out $P/ft_$n.json
  echo "$n $(python3 -c "import json;d=json.load(open(\"$P/ft_$n.json\"));print({k:d[k] for k in (\"backbone_trace_ms\",\"tower_trace_ms\",\"bb_plus_tower_trace_ms\",\"gen_ms_per_frame\")})" 2>&1 | tail -1) $(tail -1 $P/ft_$n.log) $(date +%T)"
  if ! grep -q "EXIT 0" $P/ft_$n.log; then for c in $(podman ps -q --filter ancestor=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af); do podman kill $c; done; unlock; echo "ABORT after $n"; echo CHAIN4_DONE; exit 1; fi; }
AN="DRAMA3_ADDNORM_FUSED=1"
lock


f an_sm_emm12 $AN DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=12
f an_sm_emm8kb32 $AN DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8 DRAMA3_MOE_EXPERT_MM_KB=32
f an_sm_emm8c80 $AN DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8 DRAMA3_MOE_EXPERT_MM_CORES_GU=80 DRAMA3_MOE_EXPERT_MM_CORES_DN=80
unlock
echo "io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN4_DONE
