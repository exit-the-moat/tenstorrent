source /opt/ep0/port10/bb12/env.sh
while pgrep -f "bash a13.sh" >/dev/null; do sleep 15; done
lockc 0; mkdir -p $Q/f_def; EXTRA_ENV="TT_VISIBLE_DEVICES=0" timeout 900 $D/run_gen.sh $Q/f_def.log --fast --lines all $R --out $Q/f_def; unlockc 0
echo "f_def $(tail -1 $Q/f_def.log)"
/opt/ep0/run.sh "python /opt/ep0/port5/cmp_ttfa.py $Q/f_def /opt/ep0/day4b/out_v2" 2>&1 | grep -i "identical\|frames_identical" | head -12
chk golden_on128 "DRAMA3_BB_PREFETCH=1" tests.check_paged_backbone --out $Q/golden_on128.json
grep "^\[.*trace:" $Q/golden_on128.log | cut -c1-300
echo A14_DONE
