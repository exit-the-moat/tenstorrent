#!/bin/bash
# port10 codec: t1.sh with a hard stop: on timeout, kill my bench container BEFORE releasing the card-1 lock
source /opt/ep0/port10/env10.sh
C=1; n=$1; shift; ex=$1; shift
lockc $C
EXTRA_ENV="TT_VISIBLE_DEVICES=$C $B7 $ex" timeout ${TMO:-600} $D/run_check.sh $P/$n.log "$@"
rc=$?
if [ $rc = 124 ]; then
  for id in $(podman ps --format "{{.ID}} {{.Command}}" --no-trunc | grep "bench_codec10" | grep "TT_VISIBLE_DEVICES=1" | cut -d" " -f1); do podman kill $id; done
  sleep 3; echo "TIMEOUT_KILLED" >> $P/$n.log
fi
unlockc $C
echo "$n rc=$rc $(tail -n 1 $P/$n.log) $(iochk)"
echo T1_DONE
