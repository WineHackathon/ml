# Quick start: Wine ML API on H100

This is the shortest path for the current `rirodionov-sr008` H100 deployment. It reuses the verified weights already stored on FS2 instead of downloading the 5.8 GB Git LFS bundle.

## 1. Clone the code without the bundled weights

Run this on the H100 host or another Linux host with access to the private GitHub repository:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch codex/quickstart \
  https://github.com/WineHackathon/ml.git wine-ml-api
cd wine-ml-api
```

`GIT_LFS_SKIP_SMUDGE=1` leaves the bundled model files as LFS pointers in the checkout. That is intentional for this path: the server will load real weights from FS2. Do not run `git lfs pull` when using the external model root below.

## 2. Point setup and launch at the persistent models

On `rirodionov-sr008`, the verified model root is:

```bash
export WINE_MODELS_ROOT=/workspace-SR008.fs2/rodionov/data/models/wine-ml/fc3bf115-h100-20260925/models
```

Set `HOST` to the worker's private interface address. The current worker is `10.227.91.47`; check `hostname -I` if the worker has been recreated. Bind only to a private interface, not a public one.

```bash
export HOST=10.227.91.47
export PORT=8080
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

`setup-h100.sh` creates the CUDA 12.8 PyTorch and GPU OCR environments, verifies the existing FS2 model hashes and the checked-in index, and does not pull LFS when `WINE_MODELS_ROOT` points outside the checkout. Python 3.12 and `uv` are required. Leave the launch command running; use `Ctrl-C` to stop it. For a persistent terminal, run these commands inside `tmux`.

For a persistent SSH session, start `tmux` before launching the server:

```bash
tmux new -s wine-ml-api
./scripts/run-h100.sh
```

Detach without stopping the API with `Ctrl-B`, then `D`; later reconnect with `tmux attach -t wine-ml-api`.

## 3. Wait until the model is ready

From a second shell on the host:

```bash
curl -i http://10.227.91.47:8080/health
curl -i http://10.227.91.47:8080/ready
```

`/ready` must return HTTP 200 with `{"ready":true,"error":null}` before the application sends requests. Cold startup has measured around 159–164 seconds. `/health` is liveness only.

## 4. Send the first request

```bash
curl -fS -X POST http://10.227.91.47:8080/v1/recognize \
  -F 'image=@bottle.jpg;type=image/jpeg'
```

Upload image bytes as multipart field `image` (JPEG, PNG, or WebP; max 20 MiB and 32 megapixels). The response contains a suggested `slug`, catalog `card`, up to five `top5` alternatives, raw scores, and `timings_ms`. `confidence` is `null`; scores are not calibrated probabilities. For the minimal organizer response, use `/v1/eval/predict`, which returns only `{"slug":"..."}`.

## Access from a developer laptop

The H100 address is private and not directly routed from the laptop. Open a tunnel and use its local port:

```bash
ssh -N -L 18080:10.227.91.47:8080 rirodionov-sr008
```

In another terminal, test with:

```bash
curl -f http://127.0.0.1:18080/ready
curl -f -F 'image=@bottle.jpg;type=image/jpeg' \
  http://127.0.0.1:18080/v1/recognize
```

A backend on the same private network can call `http://10.227.91.47:8080` directly. The worker IP may change on restart; confirm it with `hostname -I` and update the backend configuration. The API has no built-in authentication or CORS, so call it server-to-server and do not expose it to the public internet.

## Self-contained clone with weights

If FS2 is unavailable, clone normally (without `GIT_LFS_SKIP_SMUDGE=1`) or run `git lfs pull`, then omit `WINE_MODELS_ROOT` so models come from the repository checkout:

```bash
unset WINE_MODELS_ROOT
git lfs pull
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

This downloads about 5.8 GB of model LFS objects. The setup reassembles Qwen's verified parts locally. For the full backend request/response schema and error handling, see [`api-integration.md`](api-integration.md).
