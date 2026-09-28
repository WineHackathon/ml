#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
uv="${UV:-uv}"
python="${WINE_PYTHON_BIN:-python3.12}"
env_root="${WINE_ENV_ROOT:-$root}"
mkdir -p "$env_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$root/.python}"
h100_python="$env_root/.venv-h100/bin/python"
ocr_python="$env_root/.venv-ocr-gpu/bin/python"
WINE_MODELS_ROOT="${WINE_MODELS_ROOT:-$root/models}"
export WINE_MODELS_ROOT

if [ "$WINE_MODELS_ROOT" = "$root/models" ]; then
  git lfs pull
fi

"$uv" venv --allow-existing --python "$python" "$env_root/.venv-h100"
"$uv" pip sync --python "$h100_python" --torch-backend cu128 requirements-h100.lock
"$uv" pip install --python "$h100_python" --no-deps -e .
"$uv" venv --allow-existing --python "$python" "$env_root/.venv-ocr-gpu"
"$uv" pip install --python "$ocr_python" -r requirements-ocr-gpu.lock
"$h100_python" scripts/download_models.py --root "$WINE_MODELS_ROOT" --offline
"$h100_python" scripts/verify_artifacts.py
