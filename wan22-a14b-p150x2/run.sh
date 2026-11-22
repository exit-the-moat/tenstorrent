#!/bin/bash
# Wan 2.2 A14B (text-to-video, image-to-video, distilled 4-step) on two Tenstorrent p150a cards.
#
#   ./run.sh setup                         once: tt_dit from the image + patches/ -> ./work/tt-metal (checked)
#   ./run.sh smoke [cards...]              open each card, one matmul (default: 0 1)
#   ./run.sh t2v   [options]               Wan2.2-T2V-A14B, 40 steps, CFG (harness/gen.py)
#   ./run.sh i2v   --jobs F [options]      Wan2.2-I2V-A14B, 40 steps (harness/gen_i2v.py)
#   ./run.sh distill --jobs F [options]    lightx2v 4-step distilled I2V, reference sampler (harness/gen_distill.py)
#   ./run.sh merge-lora EXPERT LORA NAME   Lightning LoRA merged on the host in fp32 -> $WORK/merged/NAME/EXPERT
#   ./run.sh lightning --merged NAME [options]  Wan2.2-Lightning T2V (4 steps) from a merged checkpoint
#   ./run.sh svi --jobs F [options]        Stable Video Infinity 2.0 Pro on the distilled I2V (harness/gen_svi.py)
#   ./run.sh py SCRIPT [args]              any script of harness/ or check/ in the container, with the cards
#
# Environment:
#   MODELS   folder with the Hugging Face downloads (README "Weights"): wan22-a14b/, wan22-i2v-a14b/, lightx2v/,
#            lightning/, svi/ (only what the command needs)
#   CACHE    tt_dit weight cache + compiled kernels (bf8 T2V experts ~32 GB, umT5 ~11 GB, per checkpoint)
#   OUT      clips, runs.jsonl, pcc dumps (default ./out)       WORK  scratch, merged checkpoints (default ./work)
# Options for t2v / i2v / distill / lightning:
#   --res 480p|720p (default 480p)   --preset best|stock (default best)   --jobs F (a json list of {name, seed,
#   prompt[, image]}, paths inside the container: put files in $OUT and use /out/...)   --only a,b   --traced
#   --write-bps D:RATE (throttle container writes, e.g. /dev/sda:80mb)
# EP4ENV="VAR=val VAR2=val" adds container variables to any command (e.g. the SVI LoRA merge, see the README).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
H=/home/container_app_user/tt-metal
WORK=${WORK:-$HERE/work}
OUT=${OUT:-$HERE/out}
BASE_TREE=b414bc48a68226b505106e064f1f1dc16170f102   # models/tt_dit of tt-metal de59f8a = the image's copy
FINAL_TREE=678329539390e13e42b64010e1d16d31b09aa247  # the same after patches/0001-0013

die(){ echo "run.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------- the presets (every flag is here)
# best = "L4": bf8 weights (both experts resident at 480p), SDPA chunks 512x512 with bf8 q/k/v, 4 card-to-card links.
BEST=(-e EP4_DIT_WEIGHTS=bf8 -e EP4_SDPA_CHUNKS=512x512 -e EP4_SDPA_BF8=1)
# memory plan, needed for 720p and for every I2V run (no numerics change): umT5 freed after encoding, VAE and
# transformers never on the cards at the same time; the experts are swapped at the high/low-noise switch.
MEM=(-e EP4_ENC_EVICT=1 -e EP4_VAE_EXCL=1)
REF=(-e EP4_DISTILL_SAMPLER=ref)   # LightX2V reference sampler for the 4-step models (default in the runners too)

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
  git -c user.name=setup -c user.email=setup@localhost am -q --keep-cr "$HERE"/patches/*.patch || { git am --abort; die "patches did not apply"; }
  t=$(git rev-parse HEAD:models/tt_dit)
  [ "$t" = "$FINAL_TREE" ] || die "patched tree is $t, expected $FINAL_TREE"
  cd "$HERE" || die "no $HERE"
  if [ -z "${TT_DIT_SRC:-}" ] && [ ! -d "$WORK/pylib/imageio_ffmpeg" ]; then   # mp4 export (not in the image)
    podman run --rm -v "$WORK":/work --entrypoint /bin/bash "$IMAGE" \
      -c 'source $PYTHON_ENV_DIR/bin/activate && pip install -q --target /work/pylib imageio-ffmpeg==0.6.0' || die "pip install imageio-ffmpeg failed"
  fi
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

THROTTLE=()
container(){   # $1 = name, rest = command line inside the container; ENVS = extra -e flags
  [ -n "${MODELS:-}" ] && [ -d "$MODELS" ] || die "set MODELS to the folder with the Hugging Face downloads"
  [ -n "${CACHE:-}" ] || die "set CACHE to a folder for the weight cache (40+ GB free)"
  [ "$(git -C "$WORK/tt-metal" rev-parse HEAD:models/tt_dit 2>/dev/null)" = "$FINAL_TREE" ] || die "run ./run.sh setup first"
  mkdir -p "$CACHE/jit" "$OUT" "$WORK"
  local name=$1; shift
  local e; for e in ${EP4ENV:-}; do ENVS+=(-e "$e"); done   # extra variables, e.g. EP4ENV="EP4_MERGE_LORA_TAG=x ..."
  podman run --rm --name "$name" --device /dev/tenstorrent --security-opt=label=disable --ipc=host "${THROTTLE[@]}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "$WORK/tt-metal/models/tt_dit":"$H/models/tt_dit":ro -v "$HERE/harness":/ep4/harness:ro -v "$HERE/check":/ep4/check:ro \
    -v "$(cd "$MODELS" && pwd)":/models:ro -v "$(cd "$CACHE" && pwd)":/cache -v "$(cd "$OUT" && pwd)":/out -v "$(cd "$WORK" && pwd)":/work \
    -e TT_METAL_HOME="$H" -e ARCH_NAME=blackhole -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TT_DIT_CACHE_DIR=/cache -e TT_METAL_CACHE=/cache/jit "${ENVS[@]}" \
    -w /out --user root --entrypoint /bin/bash "$IMAGE" \
    -c "source \$PYTHON_ENV_DIR/bin/activate; export PYTHONPATH=$H:$H/ttnn:/ep4/harness:/ep4/check:/work/pylib; $*"
}

RES=480p; PRESET=best; JOBS=""; ONLY=""; TRACED=0; MERGED=""; EXTRA=()
parse(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --res) RES=$2; shift 2;;
      --preset) PRESET=$2; shift 2;;
      --jobs) JOBS=$2; shift 2;;
      --only) ONLY=$2; shift 2;;
      --traced) TRACED=1; shift;;
      --merged) MERGED=$2; shift 2;;
      --write-bps) THROTTLE=(--device-write-bps "$2"); shift 2;;
      *) EXTRA+=("$1"); shift;;
    esac
  done
  case "$RES" in 480p|720p) ;; *) die "--res is 480p or 720p";; esac
  case "$PRESET" in best|stock) ;; *) die "--preset is best or stock";; esac
  if [ -n "$JOBS" ]; then
    [ -f "$JOBS" ] || die "--jobs: no file $JOBS"
    mkdir -p "$OUT"; cp "$JOBS" "$OUT/.jobs.json"; JOBS="--jobs /out/.jobs.json"
  fi
  [ -n "$ONLY" ] && ONLY="--only $ONLY"
}

run_t2v(){
  parse "$@"; local tag="t2v_${RES}_${PRESET}"
  ENVS=(); local dl=1 links=2
  if [ "$PRESET" = best ]; then ENVS+=("${BEST[@]}"); links=4; [ "$RES" = 480p ] && dl=0; fi
  [ "$RES" = 720p ] && ENVS+=("${MEM[@]}")
  container ep4-t2v "python -u /ep4/harness/gen.py --run $tag --res $RES ${JOBS:---jobs /ep4/harness/jobs_rep2.json} $ONLY --dynamic-load $dl --links $links --traced $TRACED ${EXTRA[*]:-}"
}
run_i2v(){   # $1 = runner; I2V always needs the memory plan and the expert swap
  local runner=$1; shift; parse "$@"
  [ -n "$JOBS" ] || die "--jobs is required (json list of {name, seed, prompt, image}, image as /out/<file>)"
  ENVS=("${BEST[@]}" "${MEM[@]}")
  [ "$runner" != gen_i2v ] && ENVS+=("${REF[@]}")
  container ep4-$runner "python -u /ep4/harness/$runner.py --run ${runner}_$RES $( [ "$runner" = gen_svi ] || echo "--res $RES" ) $JOBS $ONLY --dynamic-load 1 --links 4 ${EXTRA[*]:-}"
}
run_lightning(){
  parse "$@"
  [ -n "$MERGED" ] && [ -d "$WORK/merged/$MERGED" ] || die "--merged NAME: no $WORK/merged/NAME (see ./run.sh merge-lora)"
  ENVS=("${BEST[@]}" "${REF[@]}" -e EP4_CKPT=/work/merged/$MERGED)
  local dl=0; [ "$RES" = 720p ] && { ENVS+=("${MEM[@]}"); dl=1; }
  container ep4-lightning "python -u /ep4/harness/gen_lightning.py --run lightning_$RES --no-runtime-lora --res $RES ${JOBS:---jobs /ep4/harness/jobs_rep2.json} $ONLY --dynamic-load $dl --links 4 --traced 0 ${EXTRA[*]:-}"
}
merge_lora(){   # EXPERT (transformer | transformer_2), LORA (file below $MODELS, e.g. lightning/.../high_noise_model.safetensors), NAME
  [ $# -eq 3 ] || die "merge-lora EXPERT LORA NAME"
  local ex=$1 lora=$2 name=$3
  mkdir -p "$WORK/merged/$name"
  # the other parts of the checkpoint are linked from the base download (relative to the container's /models)
  for p in model_index.json scheduler text_encoder tokenizer vae; do   # both experts get merged: run it twice
    [ -e "$WORK/merged/$name/$p" ] || ln -s "/models/wan22-a14b/$p" "$WORK/merged/$name/$p"
  done
  ENVS=()
  container ep4-merge "python -u /ep4/harness/merge_lora_stream.py --src /models/wan22-a14b/$ex --dst /work/merged/$name/$ex --lora /models/$lora"
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case "$cmd" in
  setup) setup;;
  smoke) smoke "$@";;
  t2v) run_t2v "$@";;
  i2v) run_i2v gen_i2v "$@";;
  distill) run_i2v gen_distill "$@";;
  svi) run_i2v gen_svi "$@";;
  lightning) run_lightning "$@";;
  merge-lora) merge_lora "$@";;
  py) [ $# -ge 1 ] || die "py SCRIPT [args]"; s=$1; shift; ENVS=()
      f=$( [ -f "$HERE/harness/$s" ] && echo /ep4/harness/$s || echo /ep4/check/$s ); container ep4-py "python -u $f $*";;
  -h|--help|"") sed -n '2,26p' "$0";;
  *) die "unknown command $cmd (see --help)";;
esac
