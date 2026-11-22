#!/usr/bin/env bash
# Pod entrypoint: Episode 4 NVIDIA reference, Wan2.2-T2V-A14B stock diffusers bf16 on one rented GPU.
# Status in /root/pub/status.txt; results in /root/out (copied back by launch.py). Inputs arrive by scp in /root/in.
set -u
export HOME=/root PATH=/usr/local/cuda/bin:$PATH
mkdir -p /root/pub /root/out /root/in
S=/root/pub/status.txt
st() { echo "$(date -u +%H:%M:%S) $*" >> $S; }
st "BOOT"
(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq openssh-server curl ffmpeg > /dev/null
  mkdir -p /root/.ssh /run/sshd
  echo "$PUBKEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd && echo SSHD_UP
) > /root/pub/ssh.log 2>&1 &

export HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
python3 -m pip install -q "diffusers==0.38.0" transformers accelerate imageio imageio-ffmpeg ftfy huggingface_hub \
  > /root/out/pip.log 2>&1
st "pip done rc=$?"
T0=$(date +%s)
python3 - > /root/out/download.log 2>&1 <<'PY'
import time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("Wan-AI/Wan2.2-T2V-A14B-Diffusers", revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
                      local_dir="/root/model", ignore_patterns=["assets/*"])
print("model", p, round(time.time() - t, 1), "DL_DONE", flush=True)
PY
st "download rc=$? $(( $(date +%s) - T0 )) s $(du -sh /root/model | cut -f1)"
{
  echo "## nvidia-smi"; nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap --format=csv
  echo "## python"; python3 -c "import sys,torch;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda);print(torch.cuda.get_device_name(0))"
  echo "## packages"; python3 -m pip list 2>/dev/null | grep -i -E "^(diffusers|transformers|torch|accelerate|huggingface)"
  echo "## cpu"; nproc; grep -m1 "model name" /proc/cpuinfo; free -g
} > /root/out/versions.txt 2>&1
st "versions: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

while [ ! -f /root/in/READY ]; do sleep 5; done
st "inputs ready"
cd /root/in
# power log for the whole run (W, clocks, temp every 5 s)
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -l 5 \
  > /root/out/power.csv 2>&1 &
run() {  # label prompts res
  st "start $1"
  python3 ref_wan22_gpu.py --ckpt /root/model --rev main --prompts "$2" --res "$3" --out "/root/out/$1" \
    > "/root/out/$1.log" 2>&1
  st "end $1 rc=$? $(tail -c 300 /root/out/$1/timing.jsonl 2>/dev/null | tail -1 | cut -c1-160)"
}
run r480 cats2.json 480p
run r720 cats2.json 720p
run h480 prompts8.json 480p
st "DONE"
sleep infinity
