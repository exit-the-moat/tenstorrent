#!/bin/bash
# LTX-2.5 distilled (video + audio; text-to-video and image-to-video) on two Tenstorrent p150a cards.
#
#   ./run.sh setup                   once: tt_dit from the image + ../wan22-a14b-p150x2/patches + patches/ (checked)
#   ./run.sh aux                     once: the small VAE + audio checkpoint the loaders read ($CACHE/ltx-2.5-aux-conv.safetensors)
#   ./run.sh smoke [cards...]        open each card, one matmul (default: 0 1)
#   ./run.sh gen [options]           clips with harness/gen_ltx.py (default: the three counted prompts, 720p)
#   ./run.sh py SCRIPT [args]        any script of harness/ or check/ in the container, with the cards
#
# Environment:
#   MODELS   folder that holds ltx25/ = the Lightricks/LTX-2.5 download (README "Weights")
#   CACHE    tt_dit weight cache + compiled kernels + the aux checkpoint (~25 GB)
#   OUT      clips and runs.jsonl (default ./out)      WORK  scratch, preprocessed start images (default ./work)
# Options for gen:
#   --res 480p|720p|1080p (832x448 / 1280x704 / 1920x1088; heights are multiples of 64)   --frames N (default 121 = 5 s)
#   --jobs F   json list of {name, seed, prompt[, image, strength, segments, overlap, single_out]} (default harness/jobs3.json);
#              start images are folders made by harness/prep_img.py, given as /work/<folder>
#   --repeat N (warm repeats of the list)   --long (the 249-frame single-pass memory switches, 1080p I2V)
#   --write-bps D:RATE (throttle container writes, e.g. /dev/sda:80mb)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
H=/home/container_app_user/tt-metal
WORK=${WORK:-$HERE/work}
OUT=${OUT:-$HERE/out}
BASE_TREE=b414bc48a68226b505106e064f1f1dc16170f102   # models/tt_dit of tt-metal de59f8a = the image's copy
WAN_TREE=678329539390e13e42b64010e1d16d31b09aa247    # after ../wan22-a14b-p150x2/patches (the LTX work started there)
FINAL_TREE=53283b7b96a5169316832e58af0df84d90a010c8  # after patches/0001-0012

die(){ echo "run.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------- the final preset ("best"; every flag is here)
# bf8 DiT weights (attention to_out stays bf16), the LTX-2.3 conv3d blockings borrowed for our 1x2 mesh, Gemma 4 on the
# cards in bf8, 4 card-to-card links (gen_ltx --links 4), zero-padded upsampler input with masks.
BEST=(-e LTX_DIT_WEIGHTS=bf8 -e LTX_CONV3D_BORROW=1 -e LTX_GEMMA_WEIGHTS=bf8 -e LTX_EMBED_TAG=pub -e LTX_UPS_PAD=mask
      -e LTX_AUX_CHECKPOINT=/cache/ltx-2.5-aux-conv.safetensors
      -e LTX_UPSAMPLER_PATH=/models/ltx25/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)
TILES_1080=(-e LTX_VAE_WTILES=4)                       # 1080p: the VAE decodes in 4 width tiles (8 for --long)
LONG=(-e LTX_CCL_FREE=1 -e LTX_VAE_WTILES=8)           # 249-frame single pass at 1080p (two-row AdaLN switches on by itself)

setup(){
  command -v git >/dev/null || die "git is needed for setup"
  local src="$WORK/tt-metal"
  if [ -d "$src/.git" ] && [ "$(git -C "$src" rev-parse HEAD:models/tt_dit 2>/dev/null)" = "$FINAL_TREE" ]; then
    echo "SETUP_OK (already set up: $src/models/tt_dit, tree $FINAL_TREE)"; return 0
  fi
  rm -rf "$src"; mkdir -p "$src/models"
  if [ -n "${TT_DIT_SRC:-}" ]; then     # optional: a tt-metal checkout at de59f8a instead of the image
    cp -r "$TT_DIT_SRC/models/tt_dit" "$src/models/tt_dit" || die "no $TT_DIT_SRC/models/tt_dit"
  else
    podman image exists "$IMAGE" || podman pull "$IMAGE" || die "could not pull $IMAGE"
    local cid; cid=$(podman create "$IMAGE" true) || die "podman create failed"
    podman cp "$cid:$H/models/tt_dit" "$src/models/tt_dit"; local rc=$?
    podman rm "$cid" >/dev/null
    [ $rc -eq 0 ] || die "could not copy models/tt_dit out of the image"
  fi
  find "$src/models/tt_dit" -name __pycache__ -type d -prune -exec rm -rf {} +
  cd "$src" || die "no $src"
  git init -q && git add -A models/tt_dit
  git -c user.name=setup -c user.email=setup@localhost commit -q -m "models/tt_dit of $IMAGE" || die "git commit failed"
  local t; t=$(git rev-parse HEAD:models/tt_dit)
  [ "$t" = "$BASE_TREE" ] || die "models/tt_dit is not the expected base (tree $t, expected $BASE_TREE)"
  git -c user.name=setup -c user.email=setup@localhost am -q --keep-cr "$HERE"/../wan22-a14b-p150x2/patches/*.patch || { git am --abort; die "Wan patches did not apply"; }
  t=$(git rev-parse HEAD:models/tt_dit); [ "$t" = "$WAN_TREE" ] || die "after the Wan patches the tree is $t, expected $WAN_TREE"
  git -c user.name=setup -c user.email=setup@localhost am -q --keep-cr "$HERE"/patches/*.patch || { git am --abort; die "LTX patches did not apply"; }
  t=$(git rev-parse HEAD:models/tt_dit)
  [ "$t" = "$FINAL_TREE" ] || die "patched tree is $t, expected $FINAL_TREE"
  cd "$HERE" || die "no $HERE"
  echo "SETUP_OK $(git -C "$src" rev-list --count HEAD) commits, models/tt_dit tree $t ($src)"
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

THROTTLE=(); DEV=(--device /dev/tenstorrent)
container(){   # $1 = name, rest = command line inside the container; ENVS = extra -e flags
  [ -n "${MODELS:-}" ] && [ -d "$MODELS/ltx25" ] || die "set MODELS to the folder that holds ltx25/ (the LTX-2.5 download)"
  [ -n "${CACHE:-}" ] || die "set CACHE to a folder for the weight cache (~25 GB free)"
  [ "$(git -C "$WORK/tt-metal" rev-parse HEAD:models/tt_dit 2>/dev/null)" = "$FINAL_TREE" ] || die "run ./run.sh setup first"
  mkdir -p "$CACHE/jit" "$OUT" "$WORK"
  local name=$1; shift
  podman run --rm --name "$name" "${DEV[@]}" --security-opt=label=disable --ipc=host "${THROTTLE[@]}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "$WORK/tt-metal/models/tt_dit":"$H/models/tt_dit":ro -v "$HERE/harness":/ltx/harness:ro -v "$HERE/check":/ltx/check:ro \
    -v "$(cd "$MODELS" && pwd)":/models:ro -v "$(cd "$CACHE" && pwd)":/cache -v "$(cd "$OUT" && pwd)":/out -v "$(cd "$WORK" && pwd)":/work \
    -e TT_METAL_HOME="$H" -e ARCH_NAME=blackhole -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TT_DIT_CACHE_DIR=/cache -e TT_METAL_CACHE=/cache/jit "${ENVS[@]}" \
    -w /out --user root --entrypoint /bin/bash "$IMAGE" \
    -c "source \$PYTHON_ENV_DIR/bin/activate; export PYTHONPATH=$H:$H/ttnn:/ltx/harness:/ltx/check; $*"
}

aux(){
  ENVS=(); DEV=()
  [ -f "$CACHE/ltx-2.5-aux-conv.safetensors" ] && { echo "AUX_OK (exists)"; return 0; }
  container ltx-aux "python -u /ltx/harness/mkaux.py conv /cache/ltx-2.5-aux-conv.safetensors"
}

gen(){
  local res=720p frames=121 jobs="" repeat=1 long=0 extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --res) res=$2; shift 2;;
      --frames) frames=$2; shift 2;;
      --jobs) jobs=$2; shift 2;;
      --repeat) repeat=$2; shift 2;;
      --long) long=1; shift;;
      --write-bps) THROTTLE=(--device-write-bps "$2"); shift 2;;
      *) extra+=("$1"); shift;;
    esac
  done
  local h w
  case "$res" in 480p) h=448; w=832;; 720p) h=704; w=1280;; 1080p) h=1088; w=1920;; *) die "--res is 480p, 720p or 1080p";; esac
  [ -f "$CACHE/ltx-2.5-aux-conv.safetensors" ] || die "run ./run.sh aux first"
  ENVS=("${BEST[@]}")
  [ "$res" = 1080p ] && ENVS+=("${TILES_1080[@]}")
  [ $long = 1 ] && ENVS+=("${LONG[@]}")
  local j=/ltx/harness/jobs3.json
  if [ -n "$jobs" ]; then [ -f "$jobs" ] || die "--jobs: no file $jobs"; cp "$jobs" "$OUT/.jobs.json"; j=/out/.jobs.json; fi
  container ltx-gen "python -u /ltx/harness/gen_ltx.py --run gen_$res --height $h --width $w --frames $frames --links 4 --gemma 1 --repeat $repeat --jobs $j ${extra[*]:-}"
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case "$cmd" in
  setup) setup;;
  aux) aux;;
  smoke) smoke "$@";;
  gen) gen "$@";;
  py) [ $# -ge 1 ] || die "py SCRIPT [args]"; s=$1; shift; ENVS=("${BEST[@]}")
      f=$( [ -f "$HERE/harness/$s" ] && echo /ltx/harness/$s || echo /ltx/check/$s ); container ltx-py "python -u $f $*";;
  -h|--help|"") sed -n '2,20p' "$0";;
  *) die "unknown command $cmd (see --help)";;
esac
