#!/usr/bin/env bash
# Pod entrypoint: Episode 4 NVIDIA reference, LTX-2.5 distilled (official Lightricks ltx_pipelines) on one rented GPU.
# Status in /root/pub/status.txt; results in /root/out (copied back by launch.py). Inputs arrive by scp in /root/in;
# the HF read token arrives over ssh stdin as /root/in/.hf_token and is deleted right after the download.
set -u
export HOME=/root PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
mkdir -p /root/pub /root/out /root/in
S=/root/pub/status.txt
st() { echo "$(date -u +%H:%M:%S) $*" >> $S; }
st "BOOT"
(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq openssh-server curl ffmpeg git > /dev/null
  mkdir -p /root/.ssh /run/sshd
  echo "$PUBKEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd && echo SSHD_UP
) > /root/pub/ssh.log 2>&1
st "sshd $(tail -1 /root/pub/ssh.log)"

# ---- install the official repo (same commit as the tt port's CPU reference) ----
(
  python3 -m pip install -q uv
  cd /root && git clone -q https://github.com/Lightricks/LTX-2.git && cd LTX-2 && git checkout -q f0f2f86
  git log -1 --format='%H %cd %s'
  export UV_LINK_MODE=copy
  uv sync --extra natten
) > /root/out/install.log 2>&1
st "install rc=$? $(tail -1 /root/out/install.log | cut -c1-120)"
PY=/root/LTX-2/.venv/bin/python

# ---- download (token from file, never on a command line) ----
while [ ! -f /root/in/.hf_token ]; do sleep 5; done
st "token present"
T0=$(date +%s)
HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1 $PY - > /root/out/download.log 2>&1 <<'PY'
import time
from huggingface_hub import hf_hub_download
tok = open("/root/in/.hf_token").read().strip()
files = ["diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
         "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
         "vae/ltx-2.5-video-vae-conv-bf16.safetensors", "vae/ltx-2.5-video-vae-bf16.safetensors",
         "vae/ltx-2.5-audio-vae-bf16.safetensors",
         "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
         "model_patches/ltx-2.5-duration-head-bf16.safetensors", "README.md"]
for f in files:
    t = time.time()
    for attempt in range(4):
        try:
            hf_hub_download("Lightricks/LTX-2.5", f, revision="49b093a39c19169a8a81721dcc541e07e945232f",
                            local_dir="/root/model", token=tok)
            break
        except Exception as e:
            print("retry", f, attempt, repr(e)[:200], flush=True)
    print("got", f, round(time.time() - t, 1), "s", flush=True)
print("DL_DONE", flush=True)
PY
rc=$?
shred -u /root/in/.hf_token 2>/dev/null || rm -f /root/in/.hf_token
st "download rc=$rc $(( $(date +%s) - T0 )) s $(du -sh /root/model | cut -f1) $(tail -1 /root/out/download.log) token_deleted=$([ -f /root/in/.hf_token ] && echo NO || echo yes)"
( cd /root/model && sha256sum $(find . -name '*.safetensors' | sort) ) > /root/out/model.sha256 2>&1 &

{
  echo "## nvidia-smi"; nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap --format=csv
  echo "## python"; $PY -c "import sys,torch;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print(torch.cuda.get_device_name(0))"
  echo "## ltx"; (cd /root/LTX-2 && git log -1 --format='%H %cd')
  echo "## cpu"; nproc; grep -m1 "model name" /proc/cpuinfo; free -g; df -h /root
} > /root/out/versions.txt 2>&1
(cd /root/LTX-2 && uv pip freeze --python $PY) > /root/out/pip_freeze.txt 2>&1
st "versions: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) $(grep '^torch' /root/out/versions.txt)"

while [ ! -f /root/in/READY ]; do sleep 5; done
st "inputs ready"
cd /root/in
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 500 \
  > /root/out/power.csv 2>&1 &

stopped() { [ -f /root/in/STOP ]; }
run() {  # label jobs H W frames warmup repeats retain [vae]
  stopped && { st "skip $1 (STOP)"; return; }
  st "start $1"
  $PY ltx_ref.py --jobs "$2" --height "$3" --width "$4" --frames "$5" --warmup "$6" --repeats "$7" --retain "$8" \
    --vae "${9:-conv}" --out "/root/out/$1" > "/root/out/$1.log" 2>&1
  local rc=$?
  st "end $1 rc=$rc $(grep -o '"total_s": [0-9.]*' /root/out/$1/timing.jsonl 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
}
cold() {  # label H W  -- the official CLI, one fresh process, wall time incl. imports, model loads, encode, mp4
  stopped && { st "skip $1 (STOP)"; return; }
  st "start $1"
  M=/root/model
  local t0=$(date +%s.%N)
  $PY -m ltx_pipelines.distilled \
    --transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path $M/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --height "$2" --width "$3" --num-frames 121 --frame-rate 24 --seed 101 \
    --prompt "$(python3 -c 'import json;print(json.load(open("/root/in/motion1.json"))[0]["prompt"])')" \
    --output-path /root/out/$1_motion_s101.mp4 > /root/out/$1.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"label\": \"$1\", \"rc\": $rc, \"wall_s\": $(python3 -c "print(round($t1-$t0,2))"), \"t0\": $t0, \"t1\": $t1}" >> /root/out/cold.jsonl
  st "end $1 rc=$rc wall $(python3 -c "print(round($t1-$t0,1))") s"
}
cold   cold720  704 1280
run    t720     motion1.json 704 1280 121 1 2 none
run    t1080    motion1.json 1088 1920 121 1 2 none
run    i2v1080  i2v5s.json   1088 1920 121 1 2 none
run    long249  i2vlong.json 1088 1920 249 1 1 none
run    t720gpu  motion1.json 704 1280 121 1 2 gpu
run    t1080gpu motion1.json 1088 1920 121 1 2 gpu
run    h720     harness8.json 704 1280 121 0 1 none
run    b720     brand6.json   704 1280 121 0 1 none
cold   cold1080 1088 1920
run    long377  i2vlong.json 1088 1920 377 0 1 none
run    long505  i2vlong.json 1088 1920 505 0 1 none
run    diff720  motion1.json 704 1280 121 0 1 none diff
st "DONE"
sleep infinity
