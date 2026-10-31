#!/bin/bash
# port8 gate: <name> <env>: 8-line Whisper sanity of run dir <name> (exists), seeds 1-5 close/fix run + Whisper
source /opt/ep0/port8/env8.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
n=$1; e=$2; x=${3:-$(cat $P/$n.args 2>/dev/null)}
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/port6/wcheck.py $P/$n" > $P/$n.wcheck.log 2>&1
echo "wcheck $(grep "mean match" $P/$n.wcheck.log)"
lock
EXTRA_ENV="$e" $D/run_gen.sh $P/s_$n.log --interactive --lines close,fix --seeds 1,2,3,4,5 $R --out $P/s_$n $x
unlock
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.gate_seeds $P/s_$n" > $P/s_$n.whisper.log 2>&1
echo "seeds $(grep mean_match $P/s_$n.whisper.log | cut -c1-200 | tr "\n" " ")"
echo GATE_DONE
