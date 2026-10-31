source /opt/ep0/port10/bb12/env.sh
chk psf "DRAMA3_SAMPLER_FUSED_PROFILE=1" tests.prof_sampler
grep "^A " $Q/psf.log
echo B4_DONE
