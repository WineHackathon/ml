#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
uv="${UV:-uv}"
python="${WINE_PYTHON_BIN:-python3.12}"
export WINE_MODELS_ROOT="${WINE_MODELS_ROOT:-$root/models}"

git lfs pull

"$uv" venv --allow-existing --python "$python" .venv-h100
"$uv" pip sync --python .venv-h100/bin/python --torch-backend cu128 requirements-h100.lock
"$uv" pip install --python .venv-h100/bin/python --no-deps -e .
"$uv" venv --allow-existing --python "$python" .venv-ocr-gpu
"$uv" pip install --python .venv-ocr-gpu/bin/python -r requirements-ocr-gpu.lock
.venv-h100/bin/python scripts/download_models.py --root "$WINE_MODELS_ROOT" --offline
.venv-h100/bin/python scripts/verify_artifacts.py
