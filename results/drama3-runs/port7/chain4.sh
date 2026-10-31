#!/bin/bash
source /opt/ep0/port7/env7.sh
R="--ref /opt/ep0/voice/ref_v2.wav"
lock
EXTRA_ENV="DRAMA3_FRAME_PIPE=1" $D/run_gen.sh $P/t_pipe.log --interactive --lines price,specs $R --out $P/t_pipe
EXTRA_ENV="DRAMA3_FRAME_PIPE=1 DRAMA3_PIPE_NOISE_EARLY=1" $D/run_gen.sh $P/t_pipe_ne.log --interactive --lines price,specs $R --out $P/t_pipe_ne
unlock
echo CHAIN4_DONE
