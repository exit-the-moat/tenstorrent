#!/bin/bash
source /opt/ep0/port10/env10.sh
cd /opt/ep0/port10/prefetch
rm -rf b_both2 b_off2 b_all2 f_def2
./gen.sh b_both2 'DRAMA3_BB_PREFETCH=1 DRAMA3_TOWER_PREFETCH=1' > b_both2.out 2>&1; echo "b_both2 $(head -1 b_both2.out)"
./gen.sh b_off2 '' > b_off2.out 2>&1; echo "b_off2 $(head -1 b_off2.out) $(grep -c '"frames_identical": true' b_off2.out)/8 vs i_bf8"
./gen.sh b_all2 'DRAMA3_BB_PREFETCH=1 DRAMA3_TOWER_PREFETCH=1 DRAMA3_SAMPLER_FUSED=1 DRAMA3_SAMPLER_FUSED_CORES=4' > b_all2.out 2>&1; echo "b_all2 $(head -1 b_all2.out)"
/opt/ep0/run.sh 'python /opt/ep0/port5/cmp_ttfa.py /opt/ep0/port10/prefetch/b_all2 /opt/ep0/port10/prefetch/b_all' 2>&1 | grep -c '"frames_identical": true, "wav_identical": true' | sed 's/^/b_all2 vs b_all identical lines /'
lockc 1
EXTRA_ENV='TT_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1' /opt/ep0/port10/prefetch/pfrun.sh '--out /opt/ep0/port10/prefetch/f_def2' 900 $D/run_gen.sh /opt/ep0/port10/prefetch/f_def2.log --fast --lines all $R --out /opt/ep0/port10/prefetch/f_def2
unlockc 1
/opt/ep0/run.sh 'python /opt/ep0/port5/cmp_ttfa.py /opt/ep0/port10/prefetch/f_def2 /opt/ep0/day4b/out_v2' 2>&1 | grep -c '"frames_identical": true, "wav_identical": true' | sed 's/^/f_def2 identical to day4b\/out_v2: /'
python3 /opt/ep0/port7/rtf.py b_both2 b_off2 b_all2
cd /opt/ep0/port10; for d in b_all b_both; do ln -sfn prefetch/$d pf_$d; done; ./wc.sh pf_b_all pf_b_both 2>&1 | grep wcheck
echo "$(iochk)"; echo CHAIN5_DONE
