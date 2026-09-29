# Wine ML API: backend handoff

This is the integration and operations handoff for the application backend. The service accepts image bytes and returns a proposed wine-catalog match; it does not accept an image URL or JSON/base64 payload. Send `multipart/form-data` with the field name `image`.

## Deployment endpoints

On a private network, bind the service to a private GPU-worker address and use `http://PRIVATE_GPU_HOST:8080` as the base URL. That address is deployment-specific and may change when the worker is recreated.

For external backends, use `https://akcizny-sbor.ru.tuna.am`. Send the shared key in `X-Token`; Tuna enforces HTTPS, key authentication, and a 2 requests/second rate limit.

For local testing without Tuna, keep an SSH tunnel open to the service's loopback port (replace the SSH alias with the deployment value):

```bash
ssh -N -L 18080:127.0.0.1:8080 SSH_ALIAS
```

Then use `http://127.0.0.1:18080` as the base URL.

Public Tuna request example:

```bash
export WINE_ML_API_TOKEN='load-from-your-secret-manager'
curl -fS -H "X-Token: $WINE_ML_API_TOKEN" \
  -F 'image=@bottle.jpg;type=image/jpeg' \
  https://akcizny-sbor.ru.tuna.am/v1/recognize
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

It returns only `{"slug":"catalog-product-slug"}`. The organizer script requires a non-empty slug; it does not receive the exact-versus-similar distinction. Prefer `/v1/recognize` for the application UI.

The provided evaluation script sends no authentication header. Run it on the GPU host with `curl`, `jq`, and `awk` and the direct loopback endpoint whenever possible. It can use the SSH-tunnel address above from a laptop, but photo transfer counts against the script's 10-second timeout. Do not point it at the protected Tuna URL.

## Full response

`200 OK` returns a JSON object. This is an example from the live H100 deployment; `top5` is abbreviated here. Catalog values, scores, and timings vary by image.

```json
{
  "status": "provisional_candidate",
  "degraded": false,
  "degraded_reason": null,
  "similarity_reason": null,
  "selection_reason": "reranker",
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

- `slug`: the selected catalog slug. If `status` is `similar_candidate`, this is an analogue, not an exact identification.
- `card`: pass-through catalog data for the selected `slug`. It commonly contains `slug`, `name`, `producer`, `region`, `color`, `type`, `grapes`, and `description`; fields vary by product.
- `top5`: up to five candidate records for review and diagnostics. Each has `slug`, visual `score`, `card`, and `provenance`; this pipeline also returns `reranker_score` and OCR evidence fields.
- `status` / `degraded` / `degraded_reason`: `similar_candidate` means show a related wine with a clear label; `provisional_candidate` is the normal match state; `degraded_candidate` means Qwen reranking did not complete.
- `similarity_reason`: `low_reranker_score` or `ocr_producer_conflict` for `similar_candidate`, otherwise `null`. `selection_reason` is `reranker`, `ocr_exact_name`, or `visual`. An exact OCR name may decide a close Qwen comparison; `margin` is null in that case.
- `identity` and `gallery_policy`: provenance/trust labels. Current catalog coverage is mostly provisional, so do not present a suggestion as verified identity.
- `confidence`: always `null` for this build. `raw_scores.visual_cosine`, `raw_scores.reranker`, and `margin` are uncalibrated ranking diagnostics, not probabilities. Do not display as a percent or use as a calibrated accept/reject threshold.
- `timings_ms`: server-side stage timings; `ocr` may be `null` when OCR was not run. `total` is the API's measured inference time, not network/upload time.
- `catalog_version`, `catalog_coverage`, `model_version`: diagnostics for logs/support; they can change with deployment/catalog updates.

The current catalog index covers 2,087 of 2,103 manifest products, and most identities are provisional. The sample result above demonstrates the response shape only; it is not an accuracy claim.

The similarity label uses a Qwen score floor and a clear OCR producer conflict. These heuristics were checked on three organizer control photos, not calibrated on an independent benchmark. A `similar_candidate` does not prove that the photographed wine is absent from the catalog.

## Errors and readiness

- `400`: empty, corrupt, unsupported, too-large, or over-resolution image. Body: `{"detail":"..."}`.
- `422`: request validation failed, usually because the `image` multipart field is missing.
- `503`: model/OCR not ready or inference failed. Body: `{"detail":"..."}`. Check `GET /ready`; do not route requests until it returns 200.
- `/health` is liveness only. `/ready` returns `{"ready":false,"error":"..."}` with HTTP 503 until initialization completes.

Use a 12-second or greater client timeout. Before the 2026-09-29 scoring update, the measured H100 field sample had p95 2.68 seconds and max 3.24 seconds; other small evaluation runs reached 4.25 seconds. Treat this as a historical benchmark result, not a per-request SLA. Cold startup was about 159–164 seconds; keep traffic off until `/ready` returns 200. Shared GPU load can cause much longer outliers; avoid concurrent GPU jobs during the organizer script's fixed 10-second request timeout.

Do not retry 400/422. For 503, check readiness and use bounded backoff; do not retry in a tight loop while the model is unready. Inference is read-only, so a bounded retry after a network timeout is safe but repeats the compute. Limit backend concurrency: the deployment uses one worker and a shared H100.

## Python backend

```python
import requests
import os

with open("bottle.jpg", "rb") as image:
    response = requests.post(
        "https://akcizny-sbor.ru.tuna.am/v1/recognize",
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

- Put the application checkout and Python environments on the deployment's persistent volume. Set `WINE_ENV_ROOT` to the persistent environment directory and `WINE_MODELS_ROOT` to the directory containing the pinned model files.
- For private-network calls, bind only to a private worker interface and configure the backend with `http://PRIVATE_GPU_HOST:8080`.
- Public base URL: `https://akcizny-sbor.ru.tuna.am`; requests require `X-Token` and are rate limited to 2 requests/second. Missing keys return 401; the deployed `/ready` and both prediction routes were verified with a valid key.
- The API itself has no authentication. The Tuna ingress protects the public endpoint; direct access to the private worker address has no auth. Do not expose port 8080 directly to the public internet.
- Store the shared Tuna token outside the repository, such as in the backend's secret manager. Send it as `X-Token`; do not commit the token or place it in a browser client. Because the token was previously committed, rotate it before production use: removing the file from the current tree does not erase it from Git history.
- Call it server-to-server from the backend, not directly from a browser: this service has no CORS policy or user auth.
- The optional `token` parameter in the example clients sends `X-Token` for Tuna. Direct private-network calls can omit it. The model API itself does not validate this token; Tuna does.

The repository's [optimization report](optimization-summary.md) documents the small evaluation sets and performance limits; do not present them as production-wide accuracy/SLO guarantees.
