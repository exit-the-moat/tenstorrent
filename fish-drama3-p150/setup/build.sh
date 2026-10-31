#!/bin/bash
# Build the patched tt-metal 70222e90 in /opt/ep0/tt-metal with the TT image's toolchain.
# Log markers: STEP ..., BUILD_DONE, BUILD_FAILED. Takes about 20 min (19 min measured with JOBS=8 on a Ryzen 7 5700G).
set -o pipefail
cd /opt/ep0/tt-metal || { echo "BUILD_FAILED no worktree"; exit 1; }
echo "STEP submodules $(date +%T)"
git submodule update --init --recursive --depth 1 || { echo "BUILD_FAILED submodules $(date +%T)"; exit 1; }
echo "STEP build $(date +%T)"
IMG=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af
podman run --rm --name "${BUILD_NAME:-ep0-build}" -v /opt/ep0/tt-metal:/work:Z -w /work --user root --entrypoint bash "$IMG" \
  -c 'git config --global --add safe.directory "*"; unset LD_LIBRARY_PATH PYTHONPATH; export TT_METAL_HOME=/work ARCH_NAME=blackhole; export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-10}; ionice -c3 nice -n10 ./build_metal.sh --release --build-dir build_Release' \
  2>&1 | grep --line-buffered -E '^\[[0-9]+/[0-9]+\]|error:|Error|FAILED|CMake Error' | awk '/^\[/{ if (++n % 200) next } {print; fflush()}'
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then echo "BUILD_DONE $(date +%T)"; else echo "BUILD_FAILED build rc=$rc $(date +%T)"; fi
