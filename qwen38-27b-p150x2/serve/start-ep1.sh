#!/bin/bash
# Episode 1: Qwen3.8-27B (bf16 HF weights) on 1 or 2 Tenstorrent p150a (TP=2 over the QSFP-DD links), vLLM-TT.
# Starts the stock vllm-tt-metal image as container $NAME on host port $PORT. See ../README.md, "Run it yourself".
# Env (defaults in brackets):
#   MODELS    folder that holds qwen38-bf16/ (the Hugging Face download)       [/opt/llm/hf]
#   CACHE     tensor cache, ~23 GB per mesh, written on the first start         [/opt/ep1/cache]
#   LOGDIR    where the server log goes                                         [/opt/ep1/serve]
#   MESH      P150x2 (two cards) | P150 (one card)                               [P150x2]
#   DEVS      TT_VISIBLE_DEVICES, "0,1" or "0"                                   [0,1]
#   LEN / SEQS / POOL  max context / parallel sequences / KV pool in tokens     [131072 / 8 / 262144: two cards]
#             one card: LEN=16384 SEQS=1 POOL=24576 (the 2-card values run out of DRAM; one card is batch 1 only)
#   TAG       cache subfolder                                                    [qwen38-<mesh>-default]
#   EXTRA_ENV extra podman args, "-e K=V" and "-v host:container" (the 2-card patch mount, see the README)
#   NAME / PORT  container name / host port                                      [ep1-vllm / 8085]
#   WRITE_BPS e.g. "/dev/sda:80mb" throttles the cache writes (our SATA SSD hung under big writes)  [off]
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
MODELS=${MODELS:-/opt/llm/hf}
CACHE=${CACHE:-/opt/ep1/cache}
LOGDIR=${LOGDIR:-/opt/ep1/serve}
NAME=${NAME:-ep1-vllm}
PORT=${PORT:-8085}
WRITE_BPS=${WRITE_BPS:-}
LOG=$LOGDIR/$NAME-$TAG${LOGSUFFIX:-}.log
mkdir -p "$CACHE" "$LOGDIR"; chmod 777 "$CACHE"
[ -f "$MODELS/qwen38-bf16/config.json" ] || { echo "no weights in $MODELS/qwen38-bf16 (see the README)" >&2; exit 1; }
THROTTLE=""; [ -n "$WRITE_BPS" ] && THROTTLE="--device-write-bps $WRITE_BPS"

ls /dev/tenstorrent/ >/dev/null 2>&1 || { echo "no /dev/tenstorrent" >&2; exit 1; }
MESH_DESC=""
if [ "$MESH" = "P150x2" ]; then
  MESH_DESC="-e TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto"
  FABRIC='"fabric_config": "FABRIC_1D", '
else
  FABRIC=''
fi

podman rm -f "$NAME" 2>/dev/null || true
# shellcheck disable=SC2086
podman run -d --name "$NAME" \
  --device /dev/tenstorrent --security-opt=label=disable \
  $THROTTLE \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --ipc=host -p "$PORT":8080 \
  -v "$MODELS":/models:ro \
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
nohup podman logs -f "$NAME" > "$LOG" 2>&1 &
echo "started $NAME ($MESH, devs $DEVS, tag $TAG, port $PORT); log $LOG"
echo "ready when the log shows 'Application startup complete' (first start 14-16 min: it builds the tensor cache)"
