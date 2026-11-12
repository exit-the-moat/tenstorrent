#!/bin/bash
# Qwen-Image-2512 on one or two Tenstorrent p150a cards: set up, smoke-test, make images, run a counted queue.
#
#   ./run.sh setup                                     once: tt_dit from the image + patches/ -> ./work/tt-metal (checked)
#   ./run.sh smoke [cards...]                          open each card, one matmul (default: 0 1)
#   ./run.sh gen   --model DIR --cache DIR [options]   the five test prompts once (or --prompts), images in --out
#   ./run.sh queue --model DIR --cache DIR [options]   counted queue of 20 (5 prompts x 4 seeds) after a warm-up
#   ./run.sh verify OUTDIR/TAG [split|card|split-1024|card-1024|musicvideo]   compare images byte for byte with ours
#
# Required for gen / queue:
#   --model DIR    the Hugging Face folder of Qwen/Qwen-Image-2512 (model_index.json, transformer/, text_encoder/, vae/ ...)
#   --cache DIR    tt_dit weight cache + kernel cache, written on the first start (about 29-31 GB per layout)
# Options:
#   --mode M       split (default: one model over both cards), copies (one whole model per card, two images at once),
#                  card (one card alone, --device N), guidance (each card computes one guidance branch)
#   --size S       full (default: 1920x1088, cropped to 1920x1080) or 1024 (1024x1024)
#   --prompts L    comma list from harness/prompts.json (default: keyart,text,portrait,detail,moat); "all" = every prompt
#   --prompt-file F  another prompt file in the same format (e.g. musicvideo/prompts.json); name the prompts with --prompts
#   --seed-offset N  added to every prompt's fixed seed (default 0 = our seeds)
#   --out DIR      images, runs.jsonl, queue.jsonl, logs (default ./out)
#   --tag T        sub-folder name for this run (default: <mode>[-1024] for gen, q_<mode>[-1024]_<time> for queue)
#   --device N     the card for --mode card (default 0)
#   --reps N       queue only: passes over the prompt list (default 4 = 20 images)
#   --write-bps D:RATE  throttle container writes, e.g. /dev/sda:80mb (we used it on a SATA SSD)
#   --env KEY=VAL  extra container variable (repeatable)
#
# The image, the mounts and the flags below are everything; nothing else on the host is touched.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
H=/home/container_app_user/tt-metal
WORK="$HERE/work"
BASE_TREE=b414bc48a68226b505106e064f1f1dc16170f102   # models/tt_dit of tt-metal de59f8a = the image's copy
FINAL_TREE=e180d768b98ddaf38f268f790cf2fc1d778de10b  # the same after patches/0001-0022
PROMPTS="keyart,text,portrait,detail,moat"

die(){ echo "run.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------- the final configuration (v3a8flq)
# Common to every mode: 8-bit (BF8) DiT weights, diffusers-matching guidance (norm rescale) and real-length prompts,
# text encoder and VAE on the cards, 50 steps, guidance 4, LoFi on all Linear matmuls.
COMMON="--weights bf8 --steps 50 --no-alloc --true-cfg --trim-prompt --sdpa-chunks 384x256 --center-k --encoder device --vae device --mm-fid lofi"
FUSE_SPLIT="--fuse up,addcmul,rope,attn_out,qkv,ff1,gelufast,psdpa,modcache,a8n,a8att,a8o"
FUSE_CARD="--fuse addcmul,rope,attn_out,gelufast,psdpa,modcache,a8n,a8att,a8o"
ENV_ALL=(-e TT_DIT_QWEN25VL_CENTER_K=1 -e TT_DIT_QWEN25VL_WEIGHTS=bf8 -e EP3_PSDPA_CHUNKS=320x512 -e EP3_MODCACHE=step)
ENV_CARD=(-e EP3_VAE_EVICTS_ENCODER=1)   # whole model on one card: the VAE swaps with the text encoder

setup(){
  command -v git >/dev/null || die "git is needed for setup"
  local src="$WORK/tt-metal"
  if [ -d "$src/.git" ] && [ "$(git -C "$src" rev-parse HEAD:models/tt_dit 2>/dev/null)" = "$FINAL_TREE" ]; then
    echo "SETUP_OK (already set up: $src/models/tt_dit, tree $FINAL_TREE)"; return 0
  fi
  rm -rf "$src"; mkdir -p "$src/models"
  podman image exists "$IMAGE" || podman pull "$IMAGE" || die "could not pull $IMAGE"
  local cid; cid=$(podman create "$IMAGE" true) || die "podman create failed"
  podman cp "$cid:$H/models/tt_dit" "$src/models/tt_dit"; local rc=$?
  podman rm "$cid" >/dev/null
  [ $rc -eq 0 ] || die "could not copy models/tt_dit out of the image"
  find "$src/models/tt_dit" -name __pycache__ -type d -prune -exec rm -rf {} +
  cd "$src" || die "no $src"
  git init -q && git add -A models/tt_dit
  git -c user.name=setup -c user.email=setup@localhost commit -q -m "models/tt_dit of $IMAGE" || die "git commit failed"
  local t; t=$(git rev-parse HEAD:models/tt_dit)
  [ "$t" = "$BASE_TREE" ] || die "the image's models/tt_dit is not the expected base (tree $t, expected $BASE_TREE)"
  git -c user.name=setup -c user.email=setup@localhost am -q "$HERE"/patches/*.patch || { git am --abort; die "patches did not apply"; }
  t=$(git rev-parse HEAD:models/tt_dit)
  [ "$t" = "$FINAL_TREE" ] || die "patched tree is $t, expected $FINAL_TREE"
  echo "SETUP_OK $(git rev-list --count HEAD) commits, models/tt_dit tree $t ($src)"
}

smoke(){
  local cards=("$@"); [ ${#cards[@]} -eq 0 ] && cards=(0 1)
  local rc=0 c r
  for c in "${cards[@]}"; do
    r=$(podman run --rm --device /dev/tenstorrent --security-opt=label=disable -v /dev/hugepages-1G:/dev/hugepages-1G --ipc=host \
        -v "$HERE/check/smoke.py":/smoke.py:ro -e TT_VISIBLE_DEVICES="$c" -e ARCH_NAME=blackhole -w "$H" --entrypoint /bin/bash "$IMAGE" \
        -c 'source "$PYTHON_ENV_DIR/bin/activate" && python -u /smoke.py' 2>&1 | grep -E "SMOKE|Timed out" | tail -1)
    echo "card $c: ${r:-no output}"
    echo "$r" | grep -q SMOKE_OK || rc=1
  done
  return $rc
}

verify(){
  local dir=${1:-}; local set=${2:-split}
  [ -d "$dir" ] || die "verify: no folder $dir"
  local n=0 ok=0 f want got
  while read -r want f; do
    case "$f" in "$set"/*) ;; *) continue;; esac
    local png="$dir/${f#*/}"
    [ -f "$png" ] || continue
    n=$((n + 1)); got=$(sha256sum "$png" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then ok=$((ok + 1)); echo "same   ${f#*/}"; else echo "differ ${f#*/}"; fi
  done < <(tr -d '\r' < "$HERE/reference/sha256.txt")   # a Windows checkout may add CRs
  [ $n -gt 0 ] || die "verify: none of the reference images ($set/*) found in $dir"
  echo "IDENTICAL $ok/$n"
  [ $ok -eq $n ]
}

# ---------------------------------------------------------------- gen / queue
PFILE=""; PSET=0; MODEL=""; CACHE=""; MODE=split; SIZE=full; OUT="$HERE/out"; TAG=""; DEVICE=0; REPS=4; SEEDOFF=0; THROTTLE=(); EXTRA=()
parse(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL=$2; shift 2;;
      --cache) CACHE=$2; shift 2;;
      --mode) MODE=$2; shift 2;;
      --size) SIZE=$2; shift 2;;
      --prompts) PROMPTS=$2; PSET=1; shift 2;;
      --prompt-file) PFILE=$2; shift 2;;
      --seed-offset) SEEDOFF=$2; shift 2;;
      --out) OUT=$2; shift 2;;
      --tag) TAG=$2; shift 2;;
      --device) DEVICE=$2; shift 2;;
      --reps) REPS=$2; shift 2;;
      --write-bps) THROTTLE=(--device-write-bps "$2"); shift 2;;
      --env) EXTRA+=(-e "$2"); shift 2;;
      -h|--help) sed -n '2,27p' "$0"; exit 0;;
      *) die "unknown option $1 (see --help)";;
    esac
  done
  [ -n "$MODEL" ] && [ -f "$MODEL/model_index.json" ] || die "--model must be the Qwen-Image-2512 folder (with model_index.json)"
  [ -n "$CACHE" ] || die "--cache DIR is required"
  case "$MODE" in split|copies|card|guidance) ;; *) die "--mode is split, copies, card or guidance";; esac
  case "$SIZE" in full|1024) ;; *) die "--size is full or 1024";; esac
  [ "$(git -C "$WORK/tt-metal" rev-parse HEAD:models/tt_dit 2>/dev/null)" = "$FINAL_TREE" ] || die "run ./run.sh setup first"
  if [ -n "$PFILE" ]; then
    [ -f "$PFILE" ] || die "--prompt-file: no file $PFILE"
    PFILE=$(cd "$(dirname "$PFILE")" && pwd)/$(basename "$PFILE")
    [ $PSET = 1 ] || die "--prompt-file: name the prompts to draw with --prompts (e.g. --prompts s1-01,s3-42)"
  fi
  if [ "$PROMPTS" = all ]; then
    PROMPTS=$(python3 -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["prompts"]))' "${PFILE:-$HERE/harness/prompts.json}") || die "cannot read the prompt file"
  fi
  mkdir -p "$CACHE/jit" "$OUT" || die "cannot create $CACHE or $OUT"
  MODEL=$(cd "$MODEL" && pwd); CACHE=$(cd "$CACHE" && pwd); OUT=$(cd "$OUT" && pwd)
}

NAMES=()
cleanup(){ for n in "${NAMES[@]}" ep3-copy0; do podman stop -t 30 "$n" >/dev/null 2>&1; done; }   # copy0 runs in a background subshell
trap 'echo "interrupted: stopping containers"; cleanup; exit 130' INT TERM

# one container; $1 = name, $2 = visible card ("" = both), rest = gen.py arguments
container(){
  local name=$1 vis=$2; shift 2
  local venv=(); [ -n "$vis" ] && venv=(-e TT_VISIBLE_DEVICES="$vis")
  local menv=(); [ "$MODE" != split ] && menv=("${ENV_CARD[@]}")
  local penv=(); [ -n "$PFILE" ] && penv=(-v "$PFILE":/prompts.json:ro -e EP3_PROMPTS=/prompts.json)
  NAMES+=("$name")
  podman run --rm --name "$name" --device /dev/tenstorrent --security-opt=label=disable --ipc=host "${THROTTLE[@]}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "$WORK/tt-metal/models/tt_dit":"$H/models/tt_dit":ro -v "$HERE/harness":/ep3:ro \
    -v "$MODEL":/work/qwen-image-2512:ro -v "$CACHE":/cache -v "$OUT":/out \
    -e TT_METAL_HOME="$H" -e ARCH_NAME=blackhole -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TT_DIT_CACHE_DIR=/cache -e TT_METAL_CACHE=/cache/jit -e EP3_OUT=/out \
    "${ENV_ALL[@]}" "${menv[@]}" "${venv[@]}" "${penv[@]}" "${EXTRA[@]}" \
    -w /work --user root --entrypoint /bin/bash "$IMAGE" \
    -c "source \$PYTHON_ENV_DIR/bin/activate; export PYTHONPATH=$H:$H/ttnn:/ep3; python -u /ep3/gen.py $*"
}

run_mode(){   # $1 = gen|queue
  local kind=$1 sz="" suf="" args reps warm label
  [ "$SIZE" = 1024 ] && { sz="--w 1024 --h 1024 --crop-h 1024"; suf="-1024"; }
  if [ "$kind" = queue ]; then reps=$REPS; warm=2; else reps=1; warm=0; fi
  [ -n "$TAG" ] || { if [ "$kind" = queue ]; then TAG="q_${MODE}${suf}_$(date +%Y%m%d-%H%M%S)"; else TAG="${MODE}${suf}"; fi; }
  local claim="/out/$TAG/.queue"; rm -rf "$OUT/$TAG/.queue"; mkdir -p "$OUT/$TAG/.queue" "$OUT/logs"
  args="--tag $TAG --prompts $PROMPTS $COMMON $sz --seed-offset $SEEDOFF --queue $claim --queue-reps $reps --warmup-steps $warm"
  local log="$OUT/logs/$TAG" t0; t0=$(date +%s)
  echo "$(date '+%F %T') $kind: mode $MODE, size $SIZE, prompts $PROMPTS x $reps, log $log*.log"
  local rc=0
  case "$MODE" in
    split)    container ep3-split "" --split tp --static-load --links 4 --norm-links 3 $FUSE_SPLIT $args > "$log.log" 2>&1 < /dev/null; rc=$?;;
    guidance) container ep3-guidance "" --split cfg --static-load $FUSE_CARD $args > "$log.log" 2>&1 < /dev/null; rc=$?;;
    card)     container ep3-card$DEVICE "$DEVICE" --single-card --links 1 --split tp $FUSE_CARD $args > "$log.log" 2>&1 < /dev/null; rc=$?;;
    copies)   container ep3-copy0 0 --single-card --links 1 --split tp $FUSE_CARD $args --queue-procs 2 > "${log}_card0.log" 2>&1 < /dev/null &
              local p0=$!
              # card 1 starts once card 0 has loaded: on a first start both would write the same weight cache at once
              until grep -q "pipeline ready" "${log}_card0.log" 2>/dev/null || ! kill -0 $p0 2>/dev/null; do sleep 5; done
              container ep3-copy1 1 --single-card --links 1 --split tp $FUSE_CARD $args --queue-procs 2 > "${log}_card1.log" 2>&1 < /dev/null
              local rc1=$?; wait $p0; rc=$(( $? + rc1 ));;
  esac
  echo "$(date '+%F %T') end rc=$rc after $(( $(date +%s) - t0 )) s"
  grep -hE "Traceback|TT_FATAL|TT_THROW" "$log"*.log | head -3
  [ $rc -eq 0 ] || { echo "FAILED (see $log*.log)"; return 1; }
  ls "$OUT/$TAG"/*-s*[0-9].png 2>/dev/null | sed 's/^/  /'
  if [ "$kind" = queue ]; then
    label="${MODE}${suf}"
    python3 "$HERE/harness/qsum.py" "$OUT" "$claim" "$label" "$TAG" || return 1
  else
    python3 - "$OUT/runs.jsonl" "$TAG" "$t0" <<'PY'
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    if r["tag"] == sys.argv[2] and r["t_start"] >= float(sys.argv[3]):
        print(f'  {r["prompt"]}-s{r["seed"]}: {r["seconds_total"]} s (encoder {r["seconds_encoder"]}, denoising {r["seconds_denoising"]}, VAE {r["seconds_vae"]}; {r["seconds_per_step"]} s per step)')
PY
  fi
  echo "DONE $OUT/$TAG"
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case "$cmd" in
  setup) setup;;
  smoke) smoke "$@";;
  verify) verify "$@";;
  gen) parse "$@"; run_mode gen;;
  queue) parse "$@"; run_mode queue;;
  -h|--help|"") sed -n '2,27p' "$0";;
  *) die "unknown command $cmd (setup, smoke, gen, queue, verify)";;
esac
