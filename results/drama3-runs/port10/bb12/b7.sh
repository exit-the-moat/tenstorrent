source /opt/ep0/port10/bb12/env.sh
while pgrep -f a12.sh >/dev/null; do sleep 15; done
for c in 4 8 2; do
chk csm$c "" tests.check_sampler_fused --rows 2000 --cores $c --out $Q/check_sampler_mc$c.json
grep "^{" $Q/csm$c.log | tail -1 | cut -c1-420
done
echo B7_DONE
