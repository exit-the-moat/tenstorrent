#!/bin/bash
# port10 prefetch: pfrun.sh <tag> <timeout s> <cmd...>: run cmd (it starts ONE tt-metal container whose args contain <tag>);
# on timeout: SIGABRT its python (faulthandler stack into the log), then kill that container, before the caller unlocks
tag=$1; tmo=$2; shift 2
timeout $tmo "$@"; rc=$?
if [ $rc = 124 ]; then
  for id in $(podman ps --format '{{.ID}} {{.Command}}' --no-trunc | grep -F -- "$tag" | cut -d' ' -f1); do
    pid=$(podman top $id -eo pid,args 2>/dev/null | grep -F 'python' | grep -v bash | awk '{print $1}' | head -1)
    [ -n "$pid" ] && podman exec $id kill -ABRT $pid 2>/dev/null; sleep 10; podman kill $id >/dev/null 2>&1
    echo "pfrun: TIMEOUT, killed container $id ($tag)"
  done
fi
return_rc=$rc; exit $return_rc
