# Wine ML API: backend handoff

This is the integration and operations handoff for the application backend. The service accepts image bytes and returns a proposed wine-catalog match; it does not accept an image URL or JSON/base64 payload. Send `multipart/form-data` with the field name `image`.

## Current H100 deployment

The service is running on `rirodionov-sr008`, bound to its private interface at `10.227.91.47:8080`. A backend on the same private network can use `http://10.227.91.47:8080` as its base URL. This is the current worker IP, not a stable DNS name; it may change when the worker is recreated.

For a backend outside that private network, use the protected Tuna endpoint `https://rirodionov-wine-api.ru.tuna.am`. Send the key in the `X-Token` header. Tuna enforces HTTPS, key authentication, and a 2 requests/second rate limit; the token is shared out of band and is not stored in this repository.

The address is not reachable directly from the developer laptop. For local testing, keep this tunnel open:

```bash
ssh -N -L 18080:10.227.91.47:8080 rirodionov-sr008
```

Then use `http://127.0.0.1:18080` as the base URL. Verified through this tunnel: `/ready` returned 200 and `/v1/eval/predict` returned a slug for an uploaded catalog image.

Public Tuna request example:

```bash
export WINE_ML_API_TOKEN='load-from-your-secret-manager'
curl -fS -H "X-Token: $WINE_ML_API_TOKEN" \
  -F 'image=@bottle.jpg;type=image/jpeg' \
  https://rirodionov-wine-api.ru.tuna.am/v1/recognize
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process is alive; does not guarantee models are loaded. |
| `GET` | `/ready` | Models, index, references, and OCR worker are ready. Wait for HTTP 200 before routing traffic; startup can take a few minutes. |
| `GET` | `/docs` | Interactive Swagger/OpenAPI docs exposed by FastAPI. Keep private with the API. |
| `GET` | `/openapi.json` | Machine-readable OpenAPI schema. |
| `POST` | `/v1/recognize` | Full response for application/backend use. |
| `POST` | `/v1/eval/predict` | Organizer-compatible minimal response: `{"slug":"..."}`. |

## Request

```bash
curl -fS -X POST http://HOST:8080/v1/recognize \
  -F 'image=@/path/to/bottle.jpg;type=image/jpeg'
```

Use JPEG, PNG, or WebP. The upload limit is 20 MiB and 32 megapixels. Do not send base64 JSON or a URL. The MIME type is optional; the image is decoded from the uploaded bytes. The API does not retain the upload as a catalog asset.

The short compatibility endpoint is:

```bash
curl -fS -X POST http://HOST:8080/v1/eval/predict \
  -F 'image=@/path/to/bottle.jpg;type=image/jpeg'
```

It returns only `{"slug":"catalog-product-slug"}`. Prefer `/v1/recognize` if the product UI needs a card, alternatives, or diagnostic timings.

## Full response

`200 OK` returns a JSON object. This is an example from the live H100 deployment; `top5` is abbreviated here. Catalog values, scores, and timings vary by image.

```json
{
  "status": "provisional_candidate",
  "degraded": false,
  "degraded_reason": null,
  "slug": "valeriy-zaharin-rubedo-reserve-merlo-krasnoe-suhoe-13",
  "card": {
    "slug": "valeriy-zaharin-rubedo-reserve-merlo-krasnoe-suhoe-13",
    "name": "Rubedo. Reserve",
    "producer": "Валерий Захарьин",
    "region": "Крым",
    "color": "Глубокий рубиновый",
    "type": "Красное",
    "grapes": "Мерло",
    "description": "Вкус: С тонами вишни, черешни, шелковицы, ежевики, какао, гвоздики, с мягкими танинами и долгим послевкусием."
  },
  "confidence": null,
  "uncalibrated": true,
  "raw_scores": {
    "visual_cosine": 0.897,
    "reranker": 0.695
  },
  "margin": 0.023,
  "identity": {"identity_status": "provisional", "join_method": "dual_key_candidate"},
  "top5": [
    {
      "slug": "valeriy-zaharin-rubedo-reserve-merlo-krasnoe-suhoe-13",
      "score": 0.897,
      "card": {"name": "Rubedo. Reserve"},
      "provenance": {"identity_status": "provisional", "join_method": "dual_key_candidate"},
      "reranker_score": 0.695,
      "ocr_entity_score": 0.163,
      "ocr_year_match": false,
      "ocr_year_mismatch": false
    }
  ],
  "timings_ms": {
    "decode": 62.8,
    "embedding": 707.27,
    "search": 7.41,
    "ocr": 1275.0,
    "qwen": 1416.27,
    "rerank": 1772.95,
    "total": 2550.43
  },
  "catalog_version": "be1865f14c13b25a",
  "gallery_policy": "provisional",
  "catalog_coverage": {
    "indexed_slugs": 2087,
    "manifest_records": 2103,
    "ratio": 0.9924,
    "identity_status": {"confirmed": 196, "provisional": 1891}
  },
  "model_version": {
    "source": "google/siglip2-base-patch16-384",
    "revision": "f775b65a79762255128c981547af89addcfe0f88",
    "reranker": "Qwen/Qwen3-VL-Reranker-2B",
    "reranker_config_sha256": "5980e6f24598e425e754e886bda247859857cddf6d819f16c511d36bf5d09abe"
  }
}
```

### Response fields the backend should rely on

- `slug`: the current best candidate slug. It is a suggestion, not a guaranteed identification.
- `card`: pass-through catalog data for the selected `slug`. It commonly contains `slug`, `name`, `producer`, `region`, `color`, `type`, `grapes`, and `description`; fields vary by product. Render tolerantly and do not assume every field is present.
- `top5`: up to five candidate records for a user-selection/review flow. Each has `slug`, visual `score`, `card`, and `provenance`; this pipeline also returns `reranker_score` and OCR evidence fields.
- `status` / `degraded` / `degraded_reason`: candidate state. `provisional_candidate` is the normal current state; `degraded_candidate` means a candidate is returned without completed Qwen reranking.
- `identity` and `gallery_policy`: provenance/trust labels. Current catalog coverage is mostly provisional; do not present the suggestion as verified identity.
- `confidence`: always `null` for this build. `raw_scores.visual_cosine`, `raw_scores.reranker`, and `margin` are uncalibrated ranking diagnostics, not probabilities. Do not display as a percent or use as a calibrated accept/reject threshold.
- `timings_ms`: server-side stage timings; `ocr` may be `null` when OCR was not run. `total` is the API's measured inference time, not network/upload time.
- `catalog_version`, `catalog_coverage`, `model_version`: diagnostics for logs/support; they can change with deployment/catalog updates.

The current catalog index covers 2,087 of 2,103 manifest products, and most identities are provisional. The sample result above demonstrates the response shape only; it is not an accuracy claim.

## Errors and readiness

- `400`: empty, corrupt, unsupported, too-large, or over-resolution image. Body: `{"detail":"..."}`.
- `422`: request validation failed, usually because the `image` multipart field is missing.
- `503`: model/OCR not ready or inference failed. Body: `{"detail":"..."}`. Check `GET /ready`; do not route requests until it returns 200.
- `/health` is liveness only. `/ready` returns `{"ready":false,"error":"..."}` with HTTP 503 until initialization completes.

Use a 12-second or greater client timeout. The measured H100 field sample had p95 2.68 seconds and max 3.24 seconds; other small evaluation runs reached 4.25 seconds. Treat this as a benchmark result, not a per-request SLA. Cold startup was about 159–164 seconds; keep traffic off until `/ready` returns 200.

Do not retry 400/422. For 503, check readiness and use bounded backoff; do not retry in a tight loop while the model is unready. Inference is read-only, so a bounded retry after a network timeout is safe but repeats the compute. Limit backend concurrency: the deployment uses one worker and a shared H100.

## Python backend

```python
import requests
import os

with open("bottle.jpg", "rb") as image:
    response = requests.post(
        "https://rirodionov-wine-api.ru.tuna.am/v1/recognize",
        files={"image": ("bottle.jpg", image, "image/jpeg")},
        headers={"X-Token": os.environ["WINE_ML_API_TOKEN"]},
        timeout=12,
    )
response.raise_for_status()
result = response.json()
print(result["slug"], result["top5"])
```

The runnable repository examples are [`examples/backend_client.py`](../examples/backend_client.py) and [`examples/backend_client.ts`](../examples/backend_client.ts). The Python example uses only the standard library; the TypeScript example uses built-in `fetch`/`FormData`.

## Deployment and security handoff

- Current host: `rirodionov-sr008`; code directory: `/home/jovyan/wine-ml-api`.
- Models are read from persistent FS2 at `/workspace-SR008.fs2/rodionov/data/models/wine-ml/fc3bf115-h100-20260925/models`; the setup verifies pinned hashes and does not copy them into the code directory.
- Current base URL for a backend with private-network routing: `http://10.227.91.47:8080`. The worker IP is ephemeral; re-check `hostname -I` after worker replacement. This route is private and does not require Tuna's token.
- Public base URL: `https://rirodionov-wine-api.ru.tuna.am`; the Tuna ingress requires `X-Token` and limits traffic to 2 requests/second. The protected ingress was verified from outside the H100 host: missing token returned 401, valid token returned `/ready` 200, and both prediction routes returned HTTP 200.
- For a developer laptop, tunnel with `ssh -N -L 18080:10.227.91.47:8080 rirodionov-sr008`, then call `http://127.0.0.1:18080`. Direct laptop access to the private IP is not routed.
- There is no authentication in the API itself. Tuna protects the public ingress with `X-Token`, HTTPS, and a rate limit; direct access to the private worker IP has no auth. Do not expose port 8080 directly to the public internet. Store the Tuna key in a backend secret manager, not source control.
- Call it server-to-server from the backend, not directly from a browser: this service has no CORS policy or user auth.
- The optional `token` parameter in the example clients sends `X-Token` for Tuna. Direct private-network calls can omit it. The model API itself does not validate this token; Tuna does.

The repository's [optimization report](optimization-summary.md) documents the small evaluation sets and performance limits; do not present them as production-wide accuracy/SLO guarantees.
