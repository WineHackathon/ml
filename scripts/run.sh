#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export WINE_INDEX="${WINE_INDEX:-$root/artifacts/catalog_v2/catalog_index.npz}"
export WINE_MANIFEST="${WINE_MANIFEST:-$root/data/catalog_manifest.jsonl}"
export WINE_MODEL="${WINE_MODEL:-$root/models/f775b65a79762255128c981547af89addcfe0f88}"
export WINE_MODEL_REVISION="${WINE_MODEL_REVISION:-f775b65a79762255128c981547af89addcfe0f88}"
export WINE_RERANKER="${WINE_RERANKER:-qwen}"
export WINE_RERANKER_MODEL="${WINE_RERANKER_MODEL:-$root/models/Qwen3-VL-Reranker-2B}"
export WINE_RERANKER_CONFIG="${WINE_RERANKER_CONFIG:-$root/artifacts/qwen_reranker/config.json}"
export WINE_OCR_PYTHON="${WINE_OCR_PYTHON:-$root/.venv-ocr/bin/python}"
export WINE_OCR_SCRIPT="${WINE_OCR_SCRIPT:-$root/scripts/ocr_rerank.py}"
export WINE_OCR_CACHE="${WINE_OCR_CACHE:-$root/artifacts/live_ocr_cache.json}"
export WINE_OCR_WORKER="${WINE_OCR_WORKER:-1}"
export WINE_LOCAL_ONLY="${WINE_LOCAL_ONLY:-1}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$root/models/paddlex}"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

python="${WINE_PYTHON:-$root/.venv/bin/python}"
if [ ! -x "$python" ]; then
  python=$(command -v python)
fi
exec "$python" -m uvicorn wine_api.api:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" --workers 1
