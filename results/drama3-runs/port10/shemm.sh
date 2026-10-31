#!/bin/bash
# port10: shared expert inside ExpertMM (DRAMA3_MOE_SHARED_IN_EMM=1) vs port8 S8, single stream, card 0 alone
source /opt/ep0/port10/env10.sh
I7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1"
S8="$P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8"
SH="DRAMA3_MOE_SHARED_IN_EMM=1"
echo "start $(date +%T) $(iochk)"
lockc 0
for n in s8 s8sh; do
  [ $n = s8sh ] && ex="$S8 $SH" || ex="$S8"
  EXTRA_ENV="TT_VISIBLE_DEVICES=0 $I7 $ex" timeout 600 $D/run_check.sh $P/ft_$n.log tests.profile_frame7 --mode time --out $P/ft_$n.json; echo "ft_$n $(tail -1 $P/ft_$n.log)"
  grep -E "TT_FATAL|Error" $P/ft_$n.log | head -2 | cut -c1-250
done
for n in s8 s8sh; do
  [ $n = s8sh ] && ex="$S8 $SH" || ex="$S8"
  EXTRA_ENV="TT_VISIBLE_DEVICES=0 $I7 $ex" timeout 600 $D/run_check.sh $P/golden_$n.log tests.check_paged_backbone --out $P/golden_$n.json; echo "golden_$n $(tail -1 $P/golden_$n.log)"
done
for r in 1 2; do for n in s8 s8sh; do
  [ $n = s8sh ] && ex="$S8 $SH" || ex="$S8"
  o=$P/r_${n}_$r; mkdir -p $o
  EXTRA_ENV="TT_VISIBLE_DEVICES=0 $ex" $D/run_gen.sh $o.log --interactive --lines all $R --out $o; echo "r_${n}_$r $(tail -1 $o.log)"
done; done
unlockc 0
echo "end $(date +%T) $(iochk)"; echo SHEMM_DONE
