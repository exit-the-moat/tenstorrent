#!/bin/bash
# Episode 1, 1 vs 2 cables: the 2-card default config of last night (start 6: gdn_tp_v5 patches, host sampling) with
# ONE QSFP-DD cable plugged in (Sebastian unplugged the other, 2026-11-01 ~11:45). Same ladder as 2card-2cable-default.
set -u
iochk() { echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1 | tr -d " ")"; }
echo "=== 1cable start $(date -Is) $(iochk)"
until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo "ep1-1cable $$" > /opt/ep0/card.lock/owner
P=/opt/ep1/patches/gdn_tp_v5.py
MD=/home/container_app_user/p150_x2_1cable.textproto
B=/home/container_app_user/tt-metal/models/demos/blackhole/qwen36/tt
EXTRA_ENV="-v /opt/ep1/serve/p150_x2_1cable_mesh_graph_descriptor.textproto:$MD:ro -e TT_MESH_GRAPH_DESC_PATH=$MD -v $P:$B/gdn/tp.py:ro -e QWEN36_GDN_CONV1D=0 -e QWEN36_GDN_PREFILL_DRAM=1 -e QWEN36_GDN_UNFUSED_RS=1" \
  LOGSUFFIX=-1cable /opt/ep1/serve/start-ep1.sh
L=/opt/ep1/serve/ep1-vllm-qwen38-p150x2-default-1cable.log
ok=0
for i in $(seq 1 360); do
  grep -q "Application startup complete" $L 2>/dev/null && { ok=1; break; }
  grep -q -E "EngineCore failed|TT_THROW|TT_FATAL|Traceback" $L 2>/dev/null && break
  [ "$(podman inspect -f '{{.State.Running}}' ep1-vllm 2>/dev/null)" = "true" ] || break
  sleep 10
done
if [ $ok = 1 ]; then
  echo "server ready $(date -Is)"
  export EP1_IMAGE=vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af EP1_TT_METAL=de59f8a+ep1-gdn-v5 EP1_POWER_CARDS=0,1
  C="--config 2card-1cable-default --cards 2 --cables 1 --precision default-bfp8-bfp4mlp --repeats 2"
  echo "PHASE ladder START $(date -Is)"
  python3 /opt/ep1/bench/bench.py $C --users 1,4,8 --inputs 1024,4096 && echo "PHASE ladder DONE $(date -Is)" || echo "PHASE ladder FAILED"
  echo "PHASE long START $(date -Is)"
  python3 /opt/ep1/bench/bench.py $C --users 1 --inputs 8192,32768 && echo "PHASE long DONE $(date -Is)" || echo "PHASE long FAILED"
else
  echo "SERVER FAILED $(date -Is)"; tail -30 $L
fi
podman stop -t 30 ep1-vllm >/dev/null 2>&1; podman rm -f ep1-vllm >/dev/null 2>&1
rm -rf /opt/ep0/card.lock
echo "=== 1cable end $(date -Is) $(iochk)"
echo ALL DONE
