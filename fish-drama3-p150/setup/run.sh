#!/bin/bash
# Run a command against the Drama 3 tt-metal build on the p150, inside the TT image's toolchain.
# Usage: /opt/ep0/run.sh <command...>   e.g. /opt/ep0/run.sh python /opt/ep0/smoke.py
# MODEL_DIR: your download of fishaudio/drama-3 (mounted read-only as /models/drama-3).
IMG=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
: "${MODEL_DIR:?set MODEL_DIR to your fishaudio/drama-3 download}"
exec podman run --rm -e TT_VISIBLE_DEVICES --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /opt/ep0:/opt/ep0:z -v "$MODEL_DIR":/models/drama-3:ro,z -w /opt/ep0/tt-metal --user root --entrypoint bash "$IMG" -c \
  "unset LD_LIBRARY_PATH; export TT_METAL_HOME=/opt/ep0/tt-metal PYTHONPATH=/opt/ep0/tt-metal:/opt/ep0/tt-metal/ttnn ARCH_NAME=blackhole; source /home/container_app_user/tt-metal/python_env/bin/activate; $*"
