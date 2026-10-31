#!/bin/bash
# port10 quality gate on the counted runs (CPU only)
cd /opt/ep0/port10
P=/opt/ep0/port10
./wc.sh fb_short_8 f_short_8 f_short_12 f_short_16 f_short_24 fb_long_8 f_long_8 f_long_16 f_long_24
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-spk:\$PYTHONPATH python /opt/ep0/port7/spk.py $P/fb_short_8 $P/f_short_8 $P/f_short_16 $P/f_short_24" > spk_s.log 2>&1; grep "^{" spk_s.log | cut -c1-300
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-spk:\$PYTHONPATH python /opt/ep0/port7/spk.py $P/fb_long_8 $P/f_long_8 $P/f_long_16 $P/f_long_24" > spk_l.log 2>&1; grep "^{" spk_l.log | cut -c1-300
/opt/ep0/run.sh "python /opt/ep0/port7/spec_cmp.py $P/fb_short_8 $P/f_short_8 $P/f_short_16 $P/f_short_24" > spec_s.log 2>&1; grep "^{" spec_s.log | cut -c1-500
/opt/ep0/run.sh "python /opt/ep0/port7/spec_cmp.py $P/fb_long_8 $P/f_long_8 $P/f_long_16 $P/f_long_24" > spec_l.log 2>&1; grep "^{" spec_l.log | cut -c1-500
echo POST_DONE
