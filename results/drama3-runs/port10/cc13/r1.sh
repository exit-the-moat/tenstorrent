#!/bin/bash
# cc13: run one drama3 test module on card 1 under its lock with the run_gen.sh --interactive env:
#   r1.sh <name> "<extra env>" <module> <args...>   -> /opt/ep0/port10/cc13/<name>.log
source /opt/ep0/port10/env10.sh
IE="DRAMA3_FIRST_WINDOW=4 DRAMA3_FIRST_WINDOW_WAIT=1 DRAMA3_PREWARM_VOICE=1 QWEN36_GDN_FUSED_PREFILL=1 QWEN36_GROUPED_SMALL_GRID=1 DRAMA3_GDN_FUSED_DECODE=all DRAMA3_GDN_FUSED_V=3 DRAMA3_MOE_FUSED_TOPK=1 DRAMA3_MOE_FUSED_COMBINE=1 DRAMA3_MOE_FUSED_SWIGLU=1 DRAMA3_ATTN_FUSED_PRE=1 DRAMA3_ATTN_FUSED_POST=1 DRAMA3_ATTN_QKV_MERGED=1 QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1 DRAMA3_CODEC_DEV_TRACE=1 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1 DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1 QWEN36_GROUPED_COMPACT=1 DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8 DRAMA3_MOE_SHARED_IN_EMM=1"
n=$1; shift; ex=$1; shift
C=1
S=/opt/ep0/port10/cc13/$n.status
lockc $C
echo "start $(date +%T) $(iochk)" > $S
EXTRA_ENV="TT_VISIBLE_DEVICES=$C CC13_TAG=$n $IE $ex" timeout ${TMO:-420} $D/run_check.sh /opt/ep0/port10/cc13/$n.log "$@"
# a timeout kills run_check.sh but not its container: stop this job's container before the card is released
for c in $(podman ps -q); do
  if podman inspect $c --format "{{.Config.Cmd}}" | grep -q "CC13_TAG=$n "; then podman kill $c > /dev/null; echo "killed container $c" >> $S; fi
done
sleep 2
unlockc $C
echo "$n $(tail -n 1 /opt/ep0/port10/cc13/$n.log) $(iochk)" >> $S
echo R1_DONE >> $S
