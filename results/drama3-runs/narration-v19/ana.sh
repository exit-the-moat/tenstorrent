#!/bin/bash
# ana.sh LINES OUTDIR...  -> runs ana.py in the TT image (CPU Whisper)
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python /opt/ep0/v19/ana.py $*"
