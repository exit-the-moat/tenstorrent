#!/bin/bash
# Set up the Drama 3 port in /opt/ep0 (the scripts expect this path; use a symlink if your disk is elsewhere).
# Needs: a p150 with tt-kmd + firmware installed, 1 GB hugepages, podman, git, and MODEL_DIR = your download of fishaudio/drama-3 (accept its licence on Hugging Face first).
#   MODEL_DIR=/path/to/drama-3 fish-drama3-p150/setup/setup.sh
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to your fishaudio/drama-3 download}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
W=/opt/ep0
mkdir -p $W/voice $W/day3 $W/models $W/ref $W/cache
cp "$HERE/setup/run.sh" "$HERE/setup/build.sh" "$HERE/setup/smoke.py" $W/
chmod +x $W/run.sh $W/build.sh

echo "== tt-metal 70222e90 + patch + port"
if [ ! -d $W/tt-metal ]; then
  git clone https://github.com/tenstorrent/tt-metal $W/tt-metal
  git -C $W/tt-metal checkout 70222e907f979825d94d74ef43e50fa9e7f8aff8
  git -C $W/tt-metal apply "$HERE/patches/tt-metal-70222e90.patch"
  cp -r "$HERE/drama3" $W/tt-metal/models/demos/blackhole/drama3
fi
$W/build.sh

echo "== SGLang-Omni (the Fish codec / DAC sources the port imports)"
[ -d $W/ref/sglang-omni ] || git clone https://github.com/sgl-project/sglang-omni $W/ref/sglang-omni
git -C $W/ref/sglang-omni checkout 1b1f897df32c0a9f5d7e5d9d66d02e5ec00d83cd

echo "== Python extras, installed with the image's Python into side folders"
$W/run.sh "pip install --no-deps --target /opt/ep0/pyext-codec descript-audio-codec==1.0.0 torchaudio==2.11.0 --extra-index-url https://download.pytorch.org/whl/cpu"
$W/run.sh "pip install --target /opt/ep0/pyext-whisper faster-whisper==1.2.1 pyyaml"
# the Whisper model for the quality check, fetched with the huggingface_hub that faster-whisper just installed
$W/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"Systran/faster-whisper-small\", local_dir=\"/opt/ep0/models/faster-whisper-small\")'"

echo "== Config shim (Drama 3's backbone as a qwen36 checkpoint view, no weight copies)"
$W/run.sh "python models/demos/blackhole/drama3/make_shim_ckpt.py /models/drama-3 /opt/ep0/drama3_shim"

echo "== Test lines"
cp "$HERE/eval/lines.yaml" $W/voice/lines.yaml

echo "== Smoke test"
$W/run.sh "python /opt/ep0/smoke.py"

cat <<'TXT'
Done. Two more things before the first line:
 1. Weights: either download the converted cache (needs the Hugging Face CLI: pip install -U "huggingface_hub[cli]")
      hf download exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150 --local-dir /opt/ep0/cache/drama3
    or skip it: the first run converts the original weights into /opt/ep0/cache/drama3 (slow, ~25 GB written).
 2. A voice (optional): `--ref none` speaks in the model's own voice. To clone one, put a 10-30 s mono WAV of a
    voice you have the right to use in /opt/ep0/voice/ref_long.wav and its exact transcript in
    /opt/ep0/voice/ref_long.txt. The Fish Audio licence forbids imitating an identifiable person without their consent.
TXT
