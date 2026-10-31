source /opt/ep0/port10/bb12/env.sh
chk psf "DRAMA3_SAMPLER_FUSED_PROFILE=1" tests.prof_sampler
grep "^A " $Q/psf.log
chk csf5 "" tests.check_sampler_fused --rows 2000 --out $Q/check_sampler_fused5.json
grep "^{" $Q/csf5.log | tail -1 | cut -c1-700
echo B5_DONE
