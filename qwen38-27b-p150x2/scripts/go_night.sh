#!/bin/bash
L=/opt/ep1/serve/ep1-vllm-qwen38-p150x2-default.log
until grep -q "Application startup complete" $L; do
  grep -q -E "Traceback|TT_THROW|TT_FATAL" $L && { echo "SERVER FAILED, not starting the sequence"; exit 1; }
  [ "$(podman inspect -f "{{.State.Running}}" ep1-vllm 2>/dev/null)" = "true" ] || { echo "SERVER GONE"; exit 1; }
  sleep 10
done
echo "server ready $(date -Is)"
exec /opt/ep1/run_night.sh
