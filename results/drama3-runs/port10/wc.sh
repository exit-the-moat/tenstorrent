#!/bin/bash
# port10: Whisper check of run dirs (short or long workload by name: *long* -> long_lines.json); wc.sh <dir>...
for d in "$@"; do
  o=/opt/ep0/port10/$d
  if [[ $d == *long* || $d == *_l* ]]; then
    nice -n 10 /opt/ep0/run.sh "LINES_FILE=/opt/ep0/port9/long_lines.json PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port9/wcheck_lf.py $o" > $o.wcheck.log 2>&1
  else
    nice -n 10 /opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $o" > $o.wcheck.log 2>&1
  fi
  echo "wcheck $d $(grep "mean match" $o.wcheck.log)"
done
echo WC_DONE
