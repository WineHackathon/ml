# Wine ML API: backend contract

The service accepts an image upload and returns a catalog candidate. It does not accept an image URL. Send the original image bytes as `multipart/form-data` with the field name `image`.

## Current H100 deployment

The service is running on `rirodionov-sr008`, bound to its private interface at `10.227.91.47:8080`. A backend on the same private network can use `http://10.227.91.47:8080` as its base URL. This is the current worker IP, not a stable DNS name; it may change when the worker is recreated.

The address is not reachable directly from the developer laptop. For local testing, keep this tunnel open:

```bash
ssh -N -L 18080:10.227.91.47:8080 rirodionov-sr008
```

Then use `http://127.0.0.1:18080` as the base URL. Verified through this tunnel: `/ready` returned 200 and `/v1/eval/predict` returned a slug for an uploaded catalog image.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process is alive; does not guarantee models are loaded. |
| `GET` | `/ready` | Models, index, references, and OCR worker are ready. Wait for HTTP 200 before routing traffic; startup can take a few minutes. |
| `POST` | `/v1/recognize` | Full response for application/backend use. |
| `POST` | `/v1/eval/predict` | Organizer-compatible minimal response: `{"slug":"..."}`. |

## Request

```bash
curl -fS -X POST http://HOST:8080/v1/recognize \
  -F 'image=@/path/to/bottle.jpg;type=image/jpeg'
```

Use JPEG, PNG, or WebP. The upload limit is 20 MiB and 32 megapixels. Do not send base64 JSON or a URL. The MIME type is optional; the image is decoded from the uploaded bytes.

## Full response

`200 OK` returns a JSON object. Example shape (catalog values and candidate count vary):

```json
{
  "status": "provisional_candidate",
  "degraded": false,
  "degraded_reason": null,
  "slug": "catalog-product-slug",
  "card": {
    "name": "Wine name",
    "producer": "Producer",
    "volume": "0.75 л"
  },
  "confidence": null,
  "uncalibrated": true,
  "raw_scores": {
    "visual_cosine": 0.82,
    "reranker": 0.91
  },
  "margin": 0.14,
  "identity": {"status": "provisional"},
  "top5": [
    {"slug": "catalog-product-slug", "score": 0.82, "card": {"name": "Wine name"}}
  ],
  "timings_ms": {
    "decode": 12.3,
    "embedding": 410.2,
    "search": 3.1,
    "ocr": 500.0,
    "qwen": 1100.0,
    "rerank": 1800.0,
    "total": 2300.0
  },
  "catalog_version": "...",
  "gallery_policy": "provisional",
  "catalog_coverage": {
    "indexed_slugs": 2087,
    "manifest_records": 2103,
    "ratio": 0.9924,
    "identity_status": {"confirmed": 196, "provisional": 1891}
  },
  "model_version": {
    "source": "google/siglip2-base-patch16-384",
    "revision": "...",
    "reranker": "Qwen/Qwen3-VL-Reranker-2B",
    "reranker_config_sha256": "..."
  }
}
```

Treat `slug` as the proposed catalog match, not a guaranteed identification. `confidence` is deliberately `null`: the returned raw model scores are uncalibrated and must not be displayed or thresholded as probabilities. Use `top5` for a review/selection flow. `status: "degraded_candidate"` means the visual candidate is returned without completed Qwen reranking; `degraded_reason` explains why.

## Errors and readiness

- `400`: empty, corrupt, unsupported, too-large, or over-resolution image. Body: `{"detail":"..."}`.
- `503`: model/OCR not ready or inference failed. Body: `{"detail":"..."}`. Check `GET /ready`; do not route requests until it returns 200.
- `/health` is liveness only. `/ready` returns `{"ready":false,"error":"..."}` with HTTP 503 until initialization completes.

Use a backend timeout of at least 12 seconds; the measured H100 sample had p95 below 3 seconds, but this is not a per-request latency guarantee. Retry only transient 503s, with bounded backoff; do not blindly retry 400s.

## Python backend

```python
import requests

with open("bottle.jpg", "rb") as image:
    response = requests.post(
        "http://HOST:8080/v1/recognize",
        files={"image": ("bottle.jpg", image, "image/jpeg")},
        timeout=12,
    )
response.raise_for_status()
result = response.json()
print(result["slug"], result["top5"])
```

The service currently has no built-in authentication. Keep the port on a trusted private network or access it through an SSH tunnel/authenticated proxy; do not expose it directly to the public internet. The examples directory also contains Python and TypeScript client snippets.
