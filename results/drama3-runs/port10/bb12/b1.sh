source /opt/ep0/port10/bb12/env.sh

chk csf3 "" tests.check_sampler_fused --rows 50 --out $Q/check_sampler_fused3.json
grep "^{" $Q/csf3.log | tail -1 | cut -c1-600
echo B1_DONE
