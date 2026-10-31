#!/bin/bash
source /opt/ep0/port10/env10.sh
cd /opt/ep0/port10/prefetch
./gen.sh g_on3 'DRAMA3_TOWER_PREFETCH=1' > g_on3.out 2>&1
lockc 1
EXTRA_ENV='TT_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1' /opt/ep0/port10/prefetch/pfrun.sh '--out /opt/ep0/port10/prefetch/f_def' 900 $D/run_gen.sh /opt/ep0/port10/prefetch/f_def.log --fast --lines all $R --out /opt/ep0/port10/prefetch/f_def
unlockc 1
/opt/ep0/run.sh 'python /opt/ep0/port5/cmp_ttfa.py /opt/ep0/port10/prefetch/f_def /opt/ep0/day4b/out_v2' 2>&1 | grep '^[a-z_]* {' > f_def.cmp
echo "f_def $(grep -c '"frames_identical": true, "wav_identical": true' f_def.cmp) of 8 identical to day4b/out_v2 $(iochk)"
echo CHAIN2_DONE
