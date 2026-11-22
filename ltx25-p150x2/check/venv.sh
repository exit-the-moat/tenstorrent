#!/bin/bash
set -e
cd "${WORK:-.}"
export UV_CACHE_DIR=/work/uvcache
uv venv -p 3.12 /work/venv
uv pip install --python /work/venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /work/venv/bin/python --no-sources -e LTX-2/packages/ltx-core -e LTX-2/packages/ltx-pipelines
/work/venv/bin/python -c "import torch,transformers,ltx_pipelines;print(torch.__version__,transformers.__version__)"
echo DONE
