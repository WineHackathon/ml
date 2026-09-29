# Wine ML API architecture

This repository provides the recognition service for a wine-catalog application. The mobile card UI and post-search features belong to the application backend/frontend; the ML service returns JSON. See [API integration](docs/api-integration.md) and the [H100 quickstart](docs/quickstart-h100.md) for the runnable contract.

## Recognition path

1. [`preprocess.py`](src/wine_api/preprocess.py) decodes uploaded JPEG, PNG, or WebP bytes, applies EXIF orientation, converts to RGB, and rejects inputs above 20 MiB or 32 megapixels. It creates whole-image and central 70% views.
2. [`retrieval.py`](src/wine_api/retrieval.py) embeds both views with frozen SigLIP2 and searches the checked-in normalized gallery by cosine similarity. Multiple reference views of one wine collapse to one catalog slug. The OCR-enabled path retrieves the visual top 20.
3. The persistent GPU OCR worker reads the label crop. [`candidate_fusion.py`](src/wine_api/candidate_fusion.py) scores producer, product name, grape, and year evidence against gallery-backed catalog rows only. It keeps a six-item pool and may replace the last visual item with one strong OCR admission. Catalog rows without an indexed reference cannot enter the reranker.
4. [`api.py`](src/wine_api/api.py) scores those candidates sequentially with frozen `Qwen/Qwen3-VL-Reranker-2B`, using the query photograph, catalog text, and reference image. Qwen normally sets the order. If OCR finds at least two distinctive exact product-name tokens, it may promote a candidate within 0.05 of the best Qwen score.
5. The result is one selected catalog card plus diagnostic top five. Weak Qwen evidence or a clear OCR producer conflict changes the full response to `status: "similar_candidate"`: the selected slug is an analogue, not a confirmed exact identification. These rules are heuristics, not calibrated probabilities.

The catalog index covers 2,087 of 2,103 manifest wines; 16 rows have no indexed reference. Most indexed identities are provisional. No model weights were fine-tuned for this build. Reproducibility hashes and the limits of the small labeled sets are in the [optimization summary](docs/optimization-summary.md).

## Service boundaries

- `POST /v1/recognize` is for the application backend. It returns the card, `status`, alternatives, raw scores, timing, and model/catalog versions. The backend should label `similar_candidate` as a similar wine rather than displaying it as the photographed product.
- `POST /v1/eval/predict` is for the organizer's script. It returns exactly one non-empty `{"slug":"..."}` and cannot convey the exact-versus-analogue distinction. Run the script next to the GPU service when possible: it has a fixed 10-second request timeout.
- `/health` is process liveness; `/ready` is model, gallery, and OCR readiness. Send inference requests only after `/ready` returns 200.
- The API itself has no authentication. Bind to loopback or a private network. The optional public Tuna ingress requires `X-Token` and limits requests to 2 per second; backend clients should keep the token outside source control.

The native H100 launcher uses one Uvicorn worker with `.venv-h100` for inference and `.venv-ocr-gpu` for OCR. Model weights may live outside the checkout through `WINE_MODELS_ROOT`; the application root and environments should live on persistent storage. The CPU Dockerfile is a compatibility path, not the measured H100 profile.
