#!/bin/bash
# after the reboot: --fast defaults (flag off) first; flag-on 8 lines only if that one built and exited 0
source /opt/ep0/port10/env10.sh
cd /opt/ep0/port10/prefetch
rm -rf f_def g_on3 g_off2
lockc 1
EXTRA_ENV='TT_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1' /opt/ep0/port10/prefetch/pfrun.sh '--out /opt/ep0/port10/prefetch/f_def' 900 $D/run_gen.sh /opt/ep0/port10/prefetch/f_def.log --fast --lines all $R --out /opt/ep0/port10/prefetch/f_def
unlockc 1
/opt/ep0/run.sh 'python /opt/ep0/port5/cmp_ttfa.py /opt/ep0/port10/prefetch/f_def /opt/ep0/day4b/out_v2' 2>&1 | grep '^[a-z_]* {' > f_def.cmp
echo "f_def $(tail -1 f_def.log) $(grep -c '"frames_identical": true, "wav_identical": true' f_def.cmp) of 8 identical to day4b/out_v2 $(iochk)"
if [ "$(tail -1 f_def.log)" != 'EXIT 0' ]; then echo 'f_def failed: no flag-on run'; echo CHAIN3_DONE; exit; fi
./gen.sh g_on3 'DRAMA3_TOWER_PREFETCH=1' > g_on3.out 2>&1; echo "g_on3 $(grep -c '"frames_identical": true' g_on3.out) identical; $(head -1 g_on3.out)"
./gen.sh g_off2 '' > g_off2.out 2>&1; echo "g_off2 $(grep -c '"frames_identical": true' g_off2.out) of 8 identical; $(head -1 g_off2.out)"
python3 /opt/ep0/port7/rtf.py g_off2 g_on3
echo CHAIN3_DONE
