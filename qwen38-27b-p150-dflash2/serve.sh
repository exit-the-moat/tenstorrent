#!/bin/bash
# Start, stop and smoke-test the Qwen3.8-27B DFlash2 server on one or two Tenstorrent p150a cards.
#
#   ./serve.sh --cards 2 --models DIR --cache DIR [options]   two cards (P150x2, two QSFP-DD cables, TP=2)
#   ./serve.sh --cards 1 --models DIR --cache DIR [options]   one card (card 0 unless --device N)
#   ./serve.sh stop [--name NAME]                             clean stop; checks the mesh close line in the log
#   ./serve.sh smoke [cards...]                               open each card, run one matmul (default: 0 1)
#
# Required for a start:
#   --models DIR   contains qwen38-bf16/ (the model) and qwen38-dflash2/ (the drafter); see README.md
#   --cache DIR    tensor cache (about 23 GB per setup); written on the first start only
# Options:
#   --work DIR     logs, kernel cache and acceptance stats (default: ./work next to this script)
#   --port N       host port on 127.0.0.1 (default 8087)
#   --name NAME    container name (default qwen38-serve)
#   --device N     the card for --cards 1 (default 0)
#   --write-bps D:RATE  throttle container writes, e.g. /dev/sda:80mb (optional; we used it on a slow SATA SSD)
#   --env KEY=VAL  extra container variable, overrides the switch of the same name (repeatable)
#   --no-wy        2 cards: without the WY block GDN verify (QWEN38_BEST=2card-nowy, the batching switches only)
#   --ep2          2 cards: exactly the Episode 2 release of 9 Nov 2026 (QWEN38_BEST=2card-ep2)
#   --wait-min N   minutes to wait for READY (default 60)
#
# The image and the mounts below are everything; nothing else on the host is touched.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
Q=/home/container_app_user/tt-metal/models/demos/blackhole/qwen36
TTM=/home/container_app_user/tt-metal
VARIANT=""; NAME=qwen38-serve; WORK="$HERE/work"; PORT=8087; DEVICE=0; WAITMIN=60; THROTTLE=""; EXTRA=(); CARDS=""; MODELS=""; CACHE=""

die(){ echo "serve.sh: $*" >&2; exit 2; }
parse(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --cards) CARDS=$2; shift 2;;
      --models) MODELS=$2; shift 2;;
      --cache) CACHE=$2; shift 2;;
      --work) WORK=$2; shift 2;;
      --port) PORT=$2; shift 2;;
      --name) NAME=$2; shift 2;;
      --device) DEVICE=$2; shift 2;;
      --write-bps) THROTTLE="--device-write-bps $2"; shift 2;;
      --wait-min) WAITMIN=$2; shift 2;;
      --env) EXTRA+=(-e "$2"); shift 2;;
      --no-wy) VARIANT=-nowy; shift;;
      --ep2) VARIANT=-ep2; shift;;
      *) die "unknown option $1";;
    esac
  done
}

smoke(){
  local cards=("$@"); [ ${#cards[@]} -eq 0 ] && cards=(0 1)
  local rc=0
  for c in "${cards[@]}"; do
    local r
    r=$(podman run --rm --device /dev/tenstorrent --security-opt=label=disable -v /dev/hugepages-1G:/dev/hugepages-1G --ipc=host \
        -v "$HERE/check/smoke.py":/smoke.py:ro -e TT_VISIBLE_DEVICES="$c" -e ARCH_NAME=blackhole -w "$TTM" --entrypoint /bin/bash "$IMAGE" \
        -c 'source "$PYTHON_ENV_DIR/bin/activate" && python -u /smoke.py' 2>&1 | grep -E "SMOKE|Timed out" | tail -1)
    echo "card $c: ${r:-no output}"
    echo "$r" | grep -q SMOKE_OK || rc=1
  done
  return $rc
}

stop(){
  local c=$NAME log
  podman ps --format '{{.Names}}' | grep -qx "$c" || { echo "$c is not running"; return 0; }
  log=$(podman inspect -f '{{range .Mounts}}{{if eq .Destination "/stats"}}{{.Source}}{{end}}{{end}}' "$c")/server.log
  local t; t=$(date +%s)
  podman stop -t 120 "$c" >/dev/null 2>&1
  echo "stopped $c in $(( $(date +%s) - t )) s"
  podman rm -f "$c" >/dev/null 2>&1
  if grep -qE "ep2-teardown: mesh closed|clean close done" "$log" 2>/dev/null; then
    echo "CLEAN_STOP $c"
  else
    echo "NO CLOSE LINE in $log: reset both cards (tt-smi -r 0 1), then ./serve.sh smoke 0 1. If a card stays wedged, soft-reboot the host."
    return 1
  fi
}

case "${1:-}" in
  stop) shift; parse "$@"; stop; exit $?;;
  smoke) shift; smoke "$@"; exit $?;;
  ""|-h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
esac

parse "$@"
[ "$CARDS" = 1 ] || [ "$CARDS" = 2 ] || die "--cards must be 1 or 2"
[ -z "$VARIANT" ] || [ "$CARDS" = 2 ] || die "--no-wy and --ep2 are 2-card options"
[ -n "$MODELS" ] && [ -d "$MODELS/qwen38-bf16" ] && [ -d "$MODELS/qwen38-dflash2" ] || die "--models must contain qwen38-bf16/ and qwen38-dflash2/ (see README.md)"
[ -n "$CACHE" ] || die "--cache DIR is required"
[ -e /dev/tenstorrent ] || die "/dev/tenstorrent not found: install tt-kmd"
[ -d /dev/hugepages-1G ] || die "/dev/hugepages-1G not found: enable 1 GB hugepages (4 per card)"
mkdir -p "$CACHE/tt_cache" "$WORK/jit" "$WORK/stats"
MODELS=$(readlink -f "$MODELS"); CACHE=$(readlink -f "$CACHE"); WORK=$(readlink -f "$WORK")
# The image runs as its own unprivileged user, which must be able to write these three folders.
chmod a+rwx "$CACHE" "$CACHE/tt_cache" "$WORK/jit" "$WORK/stats"

if [ "$CARDS" = 2 ]; then
  BEST=2card$VARIANT; TAG=qwen38-p150x2-naivebf4
  CARD_ARGS=(-e TT_VISIBLE_DEVICES=0,1 -e MESH_DEVICE=P150x2
    -e TT_MESH_GRAPH_DESC_PATH=$TTM/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto
    -v "$HERE/patches/model_config.py":$TTM/models/tt_transformers/tt/model_config.py:ro)
else
  BEST=1card; TAG=qwen38-p150-tp1
  CARD_ARGS=(-e TT_VISIBLE_DEVICES="$DEVICE" -e MESH_DEVICE=P150)
fi

CBASE=/home/container_app_user/cache_root
podman rm -f "$NAME" >/dev/null 2>&1
# shellcheck disable=SC2086
podman run -d --name "$NAME" --device /dev/tenstorrent --security-opt=label=disable $THROTTLE \
  -v /dev/hugepages-1G:/dev/hugepages-1G --ipc=host -p 127.0.0.1:"$PORT":8080 \
  -v "$MODELS":/models:ro -v "$CACHE":$CBASE -v "$WORK/jit":/jit -v "$WORK/stats":/stats \
  -v "$HERE/qwen36":$Q:ro -e QWEN36_LAZY_LOAD=1 -e TT_METAL_CACHE=/jit \
  -e ARCH_NAME=blackhole -e HF_MODEL=/models/qwen38-bf16 -e TT_CACHE_PATH=$CBASE/tt_cache/$TAG \
  -e TORCHDYNAMO_DISABLE=1 -e DF_STATS=/stats/stats.jsonl -e QWEN38_BEST=$BEST \
  "${CARD_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
  -w $TTM --entrypoint /bin/bash "$IMAGE" -c \
  'source "$PYTHON_ENV_DIR/bin/activate" && mkdir -p "$TT_CACHE_PATH" && export PYTHONPATH=/home/container_app_user/tt-metal:$PYTHONPATH && exec python -u '"$Q"'/ep2_dflash/df_server.py' >/dev/null \
  || die "podman run failed"
LOG="$WORK/stats/server.log"
nohup podman logs -f "$NAME" > "$LOG" 2>&1 &
echo "started $NAME: QWEN38_BEST=$BEST, port 127.0.0.1:$PORT, log $LOG"
for _ in $(seq 1 $(( WAITMIN * 3 ))); do
  s=$(curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  case "$s" in
    *'"ready"'*) echo "READY"; exit 0;;
    *FAILED*) echo "NOT READY (bring-up failed)"; tail -40 "$LOG"; exit 1;;
  esac
  podman ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "NOT READY (container died)"; tail -40 "$LOG"; exit 1; }
  sleep 20
done
echo "NOT READY (timeout after $WAITMIN min): see $LOG"
exit 1
