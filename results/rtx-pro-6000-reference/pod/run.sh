#!/usr/bin/env bash
# Pod entrypoint: Drama 3 reference run with the official SGLang-Omni server on one rented GPU.
# Public files (status only, never audio) go to /root/pub (served on :8000); everything else to /root/out (scp).
set -u
export HOME=/root
mkdir -p /root/pub /root/out /root/in
S=/root/pub/status.txt
st() { echo "$(date -u +%H:%M:%S) $*" >> $S; }
st "BOOT"
(cd /root/pub && nohup python3 -m http.server 8000 --directory /root/pub > /dev/null 2>&1 &)
st "driver $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) cuda_ok $(python3 -c 'import torch; torch.zeros(1).cuda(); print(torch.version.cuda, torch.cuda.get_device_name(0))' 2>&1 | tail -1)"

# SSH for control and for the private files (reference voice in, WAVs out).
(
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null
  mkdir -p /root/.ssh /run/sshd
  echo "$PUBKEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd && echo SSHD_UP
) > /root/pub/ssh.log 2>&1 &

export HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1
T_DL0=$(date +%s.%N)
python3 - > /root/out/download.log 2>&1 <<'EOF' &
import time
from huggingface_hub import snapshot_download, hf_hub_download
t = time.time()
p = snapshot_download("fishaudio/drama-3")
print("drama-3", p, round(time.time() - t, 1), flush=True)
hf_hub_download("fishaudio/s2-pro", "codec.pth", revision="1de9996b6be38b745688de084d87a5633f714e4e")
print("codec done", round(time.time() - t, 1), "DL_DONE", flush=True)
EOF
DLPID=$!

st "clone sglang-omni"
git clone -q --filter=blob:none https://github.com/sgl-project/sglang-omni.git /workspace/sglang-omni > /root/out/clone.log 2>&1
git -C /workspace/sglang-omni checkout -q 1b1f897df32c0a9f5d7e5d9d66d02e5ec00d83cd >> /root/out/clone.log 2>&1
python3 -m pip install -q --no-deps --no-build-isolation -e /workspace/sglang-omni >> /root/out/clone.log 2>&1
st "installed rc=$? commit=$(git -C /workspace/sglang-omni rev-parse HEAD)"

{
  echo "## nvidia-smi"; nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm --format=csv
  echo "## python"; python3 -c "import sys,torch;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
  echo "## packages"; python3 -m pip list 2>/dev/null | grep -i -E "^(sglang|sglang-omni|flashinfer|triton|transformers|torch|torchaudio|descript|huggingface)"
  echo "## sglang-omni commit"; git -C /workspace/sglang-omni rev-parse HEAD
  echo "## image env"; env | grep -i -E "^(CUDA_VERSION|NV_|NVIDIA_|PYTORCH)" | sort
  echo "## cpu"; nproc; grep -m1 "model name" /proc/cpuinfo; free -g
} > /root/out/versions.txt 2>&1
st "versions logged"

wait $DLPID
T_DL1=$(date +%s.%N)
st "download done $(python3 -c "print(round($T_DL1-$T_DL0))") s: $(tail -1 /root/out/download.log)"
grep -q DL_DONE /root/out/download.log || { st "FAILED download"; sleep infinity; }

# GPU sampler: epoch + nvidia-smi every 0.5 s for the whole session.
python3 -u - > /root/out/smi.csv 2>&1 <<'EOF' &
import subprocess, time, sys
q = "power.draw,memory.used,utilization.gpu,clocks.sm,temperature.gpu"
print("epoch," + q, flush=True)
p = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "500"],
                     stdout=subprocess.PIPE, text=True)
for line in p.stdout:
    print(f"{time.time():.3f},{line.strip().replace(' ', '')}", flush=True)
EOF

cd /workspace/sglang-omni
T_SV0=$(date +%s.%N)
st "server start"
nohup sgl-omni serve --model-path fishaudio/drama-3 --host 127.0.0.1 --port 30000 > /root/out/server.log 2>&1 &
SVPID=$!
for i in $(seq 1 900); do
  code=$(python3 -c "import urllib.request as u
try: print(u.urlopen(\"http://127.0.0.1:30000/health\", timeout=5).status)
except Exception as e: print(getattr(e, \"code\", 0))")
  [ "$code" = "200" ] && break
  kill -0 $SVPID 2>/dev/null || { st "FAILED server exited"; tail -40 /root/out/server.log >> $S; sleep infinity; }
  sleep 1
done
T_SV1=$(date +%s.%N)
echo "{\"download_s\": $(python3 -c "print(round($T_DL1-$T_DL0,1))"), \"server_start_to_healthy_s\": $(python3 -c "print(round($T_SV1-$T_SV0,1))")}" > /root/out/load.json
st "server healthy after $(python3 -c "print(round($T_SV1-$T_SV0,1))") s"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > /root/out/smi_apps_idle.csv 2>&1
nvidia-smi --query-gpu=memory.used,power.draw --format=csv > /root/out/smi_idle.csv 2>&1

st "waiting for /root/in/ref.wav + lines.json + bench.py"
until [ -f /root/in/READY ]; do sleep 2; done
st "bench start"
python3 -u /root/in/bench.py /root/in /root/out > /root/out/bench.log 2>&1
st "bench rc=$?"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > /root/out/smi_apps_after.csv 2>&1
st "DONE"
sleep infinity
