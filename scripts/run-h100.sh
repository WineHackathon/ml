#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${WINE_MODELS_ROOT:?set WINE_MODELS_ROOT to the external model directory}"

export CUDA_VISIBLE_DEVICES="${WINE_CUDA_DEVICE:-0}"
export WINE_DEVICE="${WINE_DEVICE:-cuda}"
export WINE_MODEL="${WINE_MODEL:-$WINE_MODELS_ROOT/f775b65a79762255128c981547af89addcfe0f88}"
export WINE_RERANKER_MODEL="${WINE_RERANKER_MODEL:-$WINE_MODELS_ROOT/Qwen3-VL-Reranker-2B}"
export WINE_RERANKER_CONFIG="${WINE_RERANKER_CONFIG:-$root/artifacts/qwen_reranker/config-h100.json}"
export WINE_RERANKER_DEVICE="${WINE_RERANKER_DEVICE:-cuda}"
export WINE_RERANKER_PRELOAD="${WINE_RERANKER_PRELOAD:-1}"
export WINE_DISABLE_CUDNN_SDP="${WINE_DISABLE_CUDNN_SDP:-1}"
export WINE_OCR_DEVICE="${WINE_OCR_DEVICE:-gpu:0}"
export WINE_OCR_PYTHON="${WINE_OCR_PYTHON:-$root/.venv-ocr-gpu/bin/python}"
export WINE_OCR_WORKER="${WINE_OCR_WORKER:-1}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$WINE_MODELS_ROOT/paddlex}"
export WINE_PYTHON="${WINE_PYTHON:-$root/.venv-h100/bin/python}"

exec "$root/scripts/run.sh"
