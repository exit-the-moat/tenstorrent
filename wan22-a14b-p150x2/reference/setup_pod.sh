#!/bin/bash
# Episode 4 NVIDIA reference: set up a fresh GPU pod (PyTorch image with CUDA) for ref_wan22_gpu.py.
# Downloads the same pinned checkpoint as the Tenstorrent runs (~126 GB; give the pod >= 200 GB of disk).
set -e
pip install -q "diffusers==0.38.0" "transformers==5.12.1" "accelerate==1.7.0" "huggingface_hub==1.16.1" \
    imageio imageio-ffmpeg ftfy sentencepiece numpy
python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("Wan-AI/Wan2.2-T2V-A14B-Diffusers", revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
                      local_dir=os.environ.get("WAN_DIR", "./wan22-a14b"), ignore_patterns=["assets/*"], max_workers=8)
print("downloaded to", p, "in", round(time.time() - t), "s")
PY
python -c "import torch, diffusers; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), diffusers.__version__)"
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv
echo SETUP_DONE
