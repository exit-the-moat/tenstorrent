#!/bin/bash
# Episode 1: Qwen3.8-27B (bf16 HF weights) on 2x Tenstorrent p150a, TP=2, QSFP-linked, for benchmarking.
# Copy of /opt/tenstorrent/start-tt-qwen38.sh with a separate identity: container "ep1-vllm", no restart policy,
# no production network alias, host port 8085, own tensor cache under /opt/ep1/cache, throttled cache writes
# (the SATA SSD has hung and corrupted files under big writes).
# Env: MESH (P150x2 | P150), DEVS (TT_VISIBLE_DEVICES, e.g. "0,1" or "0"), LEN, SEQS, POOL, TAG (cache subdir),
#      EXTRA_ENV (extra "-e K=V" args, e.g. a precision override).
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af}
MESH=${MESH:-P150x2}
DEVS=${DEVS:-0,1}
LEN=${LEN:-131072}
SEQS=${SEQS:-8}
POOL=${POOL:-262144}
TAG=${TAG:-qwen38-$(echo "$MESH" | tr A-Z a-z)-default}
EXTRA_ENV=${EXTRA_ENV:-}
# On-device sampling needs a certified 1x4 / 1x8 mesh in this build (qwen36_vllm.py raises on (1, 2)): host sampling
# unless SAMPLE_CFG is set, e.g. SAMPLE_CFG=', \"sample_on_device_mode\": \"decode_only\"'.
SAMPLE_CFG=${SAMPLE_CFG:-}
CACHE=/opt/ep1/cache
LOG=/opt/ep1/serve/ep1-vllm-$TAG${LOGSUFFIX:-}.log
mkdir -p "$CACHE"; chmod 777 "$CACHE"

ls /dev/tenstorrent/ >/dev/null 2>&1 || { echo "no /dev/tenstorrent" >&2; exit 1; }
MESH_DESC=""
if [ "$MESH" = "P150x2" ]; then
  MESH_DESC="-e TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto"
  FABRIC='"fabric_config": "FABRIC_1D", '
else
  FABRIC=''
fi

podman rm -f ep1-vllm 2>/dev/null || true
# shellcheck disable=SC2086
podman run -d --name ep1-vllm \
  --device /dev/tenstorrent --security-opt=label=disable \
  --device-write-bps /dev/sda:80mb \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --ipc=host -p 8085:8080 \
  -v /opt/llm/hf:/models:ro \
  -v "$CACHE":/home/container_app_user/cache_root \
  -e TT_VISIBLE_DEVICES="$DEVS" \
  -e VLLM_TARGET_DEVICE=tt -e TORCHDYNAMO_DISABLE=1 \
  -e VLLM_RPC_TIMEOUT=900000 -e VLLM_CONFIGURE_LOGGING=1 \
  -e ARCH_NAME=blackhole \
  -e MESH_DEVICE="$MESH" $MESH_DESC \
  -e TT_QWEN35_TEXT_VER=qwen36_blackhole \
  -e HF_MODEL=/models/qwen38-bf16 \
  -e TT_CACHE_PATH=/home/container_app_user/cache_root/tt_cache/$TAG \
  -e QWEN36_MAX_TOKENS_ALL_USERS="$POOL" \
  $EXTRA_ENV \
  --entrypoint /bin/bash "$IMAGE" -c '
    source "$PYTHON_ENV_DIR/bin/activate" && mkdir -p "$TT_CACHE_PATH" &&
    exec python -m vllm.entrypoints.openai.api_server "$@"' -- \
  --model /models/qwen38-bf16 --served-model-name qwen38 \
  --chat-template /models/qwen38-bf16/chat_template.jinja \
  --max-model-len "$LEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$LEN" \
  --block-size 64 --seed 9472 --max-log-len 32 \
  --additional-config "{\"tt\": {${FABRIC}\"trace_region_size\": 1073741824, \"l1_small_size\": 24576${SAMPLE_CFG}}}" \
  --reasoning-parser qwen3 \
  --enable-force-include-usage \
  --host 0.0.0.0 --port 8080
nohup podman logs -f ep1-vllm > "$LOG" 2>&1 &
echo "started ep1-vllm ($MESH, devs $DEVS, tag $TAG); log $LOG"
