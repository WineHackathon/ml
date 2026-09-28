# Wine ML API

Offline wine recognition for the hackathon catalog. The default launcher runs the selected pipeline: frozen SigLIP2 retrieval, label OCR candidate fusion, and sequential `Qwen/Qwen3-VL-Reranker-2B` scoring. The checked-in index covers 2,087 of 2,103 slugs; returned scores are raw, uncalibrated diagnostics, not probabilities.

## Prepare a clean clone

Python 3.12 is the tested profile. The catalog reference images are in Git and the four pinned model snapshots are bundled through Git LFS. Install Git LFS before cloning; the Qwen weight is stored as three verified parts and reassembled locally.

```bash
git lfs pull
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
python3.12 -m venv .venv-ocr
.venv-ocr/bin/pip install -r requirements-ocr.lock
.venv/bin/python scripts/download_models.py --offline
.venv/bin/python scripts/verify_artifacts.py
./scripts/run.sh
```

`download_models.py` verifies all primary weight hashes from `models.lock.json`. Without `--offline`, missing weights may be fetched from the pinned Hugging Face revisions. The split Qwen parts are concatenated into the ignored local `models/Qwen3-VL-Reranker-2B/model.safetensors` file. The bundled 2,062 reference images and checked-in index make the service runnable without the organizer archive or evaluation photos.

## H100 setup and launch

The tested GPU profile is native Linux on one NVIDIA H100, not CUDA Docker. It requires Python 3.12, `uv`, and Git LFS. The bundled models are used by default; set `WINE_MODELS_ROOT` to an external directory (for example, persistent FS2 storage) to use an existing copy instead.

```bash
export WINE_MODELS_ROOT=/path/to/persistent/models
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

For the bundled models, omit the export above and run the two scripts. `setup-h100.sh` pulls LFS objects, creates `.venv-h100` from `requirements-h100.lock` with the CUDA 12.8 PyTorch backend and `.venv-ocr-gpu` from `requirements-ocr-gpu.lock`, assembles/verifies the local weights, and verifies the index. `run-h100.sh` selects CUDA device 0 by default, the FP32 H100 reranker config, cached reference images, cuDNN SDPA disabled, and the persistent GPU OCR worker. Override the GPU with `WINE_CUDA_DEVICE`; the service still uses one Uvicorn worker.

Wait for `GET /ready` rather than a fixed sleep. Measured cold readiness was about 159–164 seconds, and the API plus OCR worker used about 7.8–8.0 GiB RSS. The 20-image H100 field sample had p50 2,112.65 ms, p95 2,677.1 ms, max 3,237.1 ms, and no HTTP failures. This establishes sample p95 below three seconds, not that every request finishes below three seconds and not a production SLO. Keep the 12-second backend timeout used by the examples.

## API and organizer contract

```bash
curl -f http://127.0.0.1:8080/health
curl -f http://127.0.0.1:8080/ready
curl -f -F image=@query.webp http://127.0.0.1:8080/v1/eval/predict
curl -f -F image=@query.webp http://127.0.0.1:8080/v1/recognize
```

`POST /v1/eval/predict` returns exactly `{"slug":"..."}` for the organizer client. `POST /v1/recognize` adds the catalog card, top five, raw visual/Qwen scores, margin, timings, model/catalog versions, provisional identity provenance, and `confidence: null`. Uploads are decoded from bytes, capped at 20 MiB and 32 megapixels, and never treated as fetchable URLs.

To run the provided organizer script without copying it into this repository:

```bash
bash "$ORGANIZER_SCRIPT" \
  --images-dir "$ORGANIZER_IMAGES" \
  --manifest "$ORGANIZER_MANIFEST" \
  --endpoint http://127.0.0.1:8080/v1/eval/predict
```

Python and TypeScript backend examples are in `examples/`; both call `POST /v1/recognize` and use a 12-second upstream timeout. The tested Apple M4 Pro MPS + CPU-OCR profile completed full requests in roughly 7.0–8.7 seconds. Run one worker because each worker duplicates model memory. Plan for at least 8 GB of model/cache disk and a 16 GB machine.

## Deployment, auth, and rollback

The service binds to `127.0.0.1:8080` by default; configure `HOST` and `PORT`. It has no built-in user authentication. Keep it on a private backend network or put it behind the application's authenticated reverse proxy; the example clients can send a bearer token to that proxy. Route traffic only after `/ready` returns 200, and stop routing on 503.

There is no silent quality fallback. For an explicit degraded visual-only rollback, restart with:

```bash
WINE_RERANKER=off ./scripts/run.sh
```

The flat organizer response remains compatible, while `/v1/recognize` reports no reranker score. The default remains the full OCR + Qwen pipeline.

## Reproducibility and limits

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/verify_artifacts.py
```

The sanitized manifest contains relative content-addressed paths and public-source provenance. The current index records its pre-quarantine source index but is not a path-only rebase (`path_rebase_only` is false): quarantining the wrong reference removed that slug's two gallery views, so the aggregate entry and embedding hashes changed. Quarantine verification confirmed that every unaffected entry and embedding row still matches the source index exactly. `scripts/verify_artifacts.py` binds the current manifest, index, hashes, and reranker configs.

Accuracy evidence and latency are summarized in `docs/optimization-summary.md`. The gallery contains 1,891 provisional and 196 confirmed identities; 16 catalog rows have no indexed reference. One verified wrong official image for the dry Citron/Chardonnay SKU was quarantined without changing its product metadata; no verified replacement was found. Do not interpret raw similarity/reranker values as confidence, and do not claim accuracy beyond the small documented evaluation sets.

## Docker

The Dockerfile is a CPU compatibility image for mounted models/assets, not the tested performance profile:

```bash
docker build -t wine-ml-api .
docker run --rm -p 8080:8080 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data/catalog_images:/app/data/catalog_images:ro" \
  wine-ml-api
```

The local Docker daemon was unavailable during publication, so this image was not built. CUDA Docker remains unverified. Native macOS MPS and the separate native Linux H100 launcher above are the tested profiles.
