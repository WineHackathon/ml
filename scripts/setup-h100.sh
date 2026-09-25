#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
uv="${UV:-uv}"
python="${WINE_PYTHON_BIN:-python3.12}"

"$uv" venv --python "$python" .venv-h100
"$uv" pip sync --python .venv-h100/bin/python --torch-backend cu128 requirements-h100.lock
"$uv" pip install --python .venv-h100/bin/python --no-deps -e .
"$uv" venv --python "$python" .venv-ocr-gpu
"$uv" pip sync --python .venv-ocr-gpu/bin/python requirements-ocr-gpu.lock
