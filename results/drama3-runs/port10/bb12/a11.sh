source /opt/ep0/port10/bb12/env.sh
BB="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=128"
SM="DRAMA3_SAMPLER_FUSED=1"
for n in i_off i_bb i_smp i_off2 i_bb2 i_smp2; do
  case $n in i_bb*) ex="$BB";; i_smp*) ex="$SM";; *) ex="";; esac
  gen $n "$ex"
  grep -o "\"rtf\": [0-9.]*" $Q/$n.log | tr "\n" " "; echo
done
echo A11_DONE
