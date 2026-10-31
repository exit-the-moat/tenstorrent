# bb12: backbone prefetch + fused sampler (card 0 only)
source /opt/ep0/port10/env10.sh
Q=/opt/ep0/port10/bb12
IA="$I7 $P7F DRAMA3_ADDNORM_FUSED=1 DRAMA3_MOE_TOPK_SOFTMAX=1 DRAMA3_MOE_EXPERT_MM=1 DRAMA3_MOE_EXPERT_MM_V=2 DRAMA3_MOE_EXPERT_MM_D=8 DRAMA3_MOE_SHARED_IN_EMM=1"
# one check under the card-0 lock: chk <name> "<extra env>" <module> <args>
chk() { local n=$1; shift; local ex=$1; shift; lockc 0; EXTRA_ENV="TT_VISIBLE_DEVICES=0 $IA $ex" timeout ${TMO:-600} $D/run_check.sh $Q/$n.log "$@"; local rc=$?; unlockc 0; echo "$n rc=$rc $(tail -n 1 $Q/$n.log) $(iochk)"; grep -E "TT_FATAL|TT_THROW|Error|error:" $Q/$n.log | grep -v FutureWarning | head -3 | cut -c1-300; }
gen() { local n=$1; shift; local ex=$1; shift; lockc 0; mkdir -p $Q/$n; EXTRA_ENV="TT_VISIBLE_DEVICES=0 $ex" timeout ${TMO:-600} $D/run_gen.sh $Q/$n.log --interactive --lines all $R --out $Q/$n "$@"; local rc=$?; unlockc 0; echo "$n rc=$rc $(tail -n 1 $Q/$n.log) $(iochk)"; }
