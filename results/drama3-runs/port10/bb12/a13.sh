source /opt/ep0/port10/bb12/env.sh
BB="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=128"
SM="DRAMA3_SAMPLER_FUSED=1 DRAMA3_SAMPLER_FUSED_CORES=4"
for n in k_off2 k_smp4b k_both2 k_off3 k_smp8 k_both3; do
  case $n in k_smp4*) ex="$SM";; k_smp8) ex="DRAMA3_SAMPLER_FUSED=1 DRAMA3_SAMPLER_FUSED_CORES=8";; k_both*) ex="$BB $SM";; *) ex="";; esac
  gen $n "$ex"
done
echo A13_DONE
