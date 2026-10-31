#!/bin/bash
# port8 item 3: ExpertMM v2 (deep weight stream) in the golden backbone check (decode ms/step + logits PCC); item 2: top-k + softmax; item 1: h in L1
source /opt/ep0/port8/env8.sh
echo "start $(date +%T) io=$(dmesg --level=err | grep -c "I/O error")"
g() { n=$1; shift; EXTRA_ENV="$I7 DRAMA3_ADDNORM_FUSED=1 $*" timeout 480 $D/run_check.sh $P/golden_$n.log tests.check_paged_backbone --out $P/golden_$n.json
  echo "$n $(grep -o "trace: .*" $P/golden_$n.log | cut -c1-200) $(tail -1 $P/golden_$n.log) $(date +%T)"
  if ! grep -q -e "^DONE" -e "EXIT 0" $P/golden_$n.log; then for c in $(podman ps -q --filter ancestor=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af); do podman kill $c; done; unlock; echo "ABORT after $n"; echo CHAIN2_DONE; exit 1; fi; }
lock
/opt/ep0/run.sh "python /opt/ep0/smoke.py" > $P/smoke_c.log 2>&1; grep SMOKE $P/smoke_c.log
g emm2_d4 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=4
g emm2_d4_c96 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=4 DRAMA3_MOE_EXPERT_MM_CORES_GU=96 DRAMA3_MOE_EXPERT_MM_CORES_DN=96
g emm2_d6_c110 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=6 DRAMA3_MOE_EXPERT_MM_CORES_GU=110 DRAMA3_MOE_EXPERT_MM_CORES_DN=110
g emm2_d4_kb32 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=4 DRAMA3_MOE_EXPERT_MM_KB=32
EXTRA_ENV="$I7 $P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_ADDNORM_H_L1=1" timeout 400 $D/run_check.sh $P/ft_anl1.log tests.profile_frame7 --mode time --out $P/ft_anl1.json; echo "ft_anl1 $(tail -1 $P/ft_anl1.log) $(date +%T)"
unlock
echo "io=$(dmesg --level=err | grep -c "I/O error")"
echo CHAIN2_DONE
