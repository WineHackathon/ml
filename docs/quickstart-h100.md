# Quick start: Wine ML API on H100

This is the shortest path for a Linux H100 deployment. Keep the checkout, tools, base Python, environments, secrets, and runtime files under one persistent application root. Model weights can remain in a separate persistent directory, avoiding the 5.8 GB Git LFS download.

## 1. Clone the code without the bundled weights

Choose an application path on the persistent volume assigned to the deployment, then clone the private repository without its LFS weights:

```bash
git lfs install
export WINE_APP_ROOT=/path/to/persistent/workspace/wine-ml-api
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch codex/quickstart \
  https://github.com/WineHackathon/ml.git "$WINE_APP_ROOT"
cd "$WINE_APP_ROOT"
```

`GIT_LFS_SKIP_SMUDGE=1` leaves the bundled model files as LFS pointers in the checkout. That is intentional for this path: the server will load real weights from FS2. Do not run `git lfs pull` when using the external model root below.

## 2. Put tools, Python, and environments on the persistent volume

Copy the installed tools into the application root and install the managed Python there:

```bash
mkdir -p .tools .python .cache/uv .private
cp "$(command -v uv)" .tools/uv
cp "$(command -v tuna)" .tools/tuna
chmod 755 .tools/uv .tools/tuna
export UV="$WINE_APP_ROOT/.tools/uv"
export UV_CACHE_DIR="$WINE_APP_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WINE_APP_ROOT/.python"
"$UV" python install 3.12.11 --install-dir "$UV_PYTHON_INSTALL_DIR"
export WINE_PYTHON_BIN="$("$UV" python find 3.12.11)"
export WINE_ENV_ROOT="$WINE_APP_ROOT"
export HOME="$WINE_APP_ROOT"
```

Set the root to the persistent directory containing the pinned model snapshots, and place the Tuna account config in the ignored `.private` directory with mode 600:

```bash
export WINE_MODELS_ROOT=/path/to/persistent/models
cp /path/to/private/tuna-cli.yml .private/tuna-cli.yml
chmod 600 .private/tuna-cli.yml
```

Bind the API to loopback; Tuna runs on the same host and forwards requests to it.

```bash
export HOST=127.0.0.1
export PORT=8080
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

`setup-h100.sh` creates/updates the CUDA 12.8 PyTorch and GPU OCR environments under `WINE_ENV_ROOT`, verifies the external model hashes and checked-in index, and does not pull LFS when `WINE_MODELS_ROOT` points outside the checkout. Python 3.12 and `uv` are required. Keep code and environments under the persistent application root. Leave the launch command running; use `Ctrl-C` to stop it. For a persistent terminal, run these commands inside `tmux`.

For a persistent SSH session, start `tmux` before launching the server:

```bash
tmux new -s wine-ml-api
./scripts/run-h100.sh
```

Detach without stopping the API with `Ctrl-B`, then `D`; later reconnect with `tmux attach -t wine-ml-api`.

## 3. Wait until the model is ready

From a second shell on the host:

```bash
curl -i http://127.0.0.1:8080/health
curl -i http://127.0.0.1:8080/ready
```

`/ready` must return HTTP 200 with `{"ready":true,"error":null}` before the application sends requests. Cold startup has measured around 159–164 seconds. `/health` is liveness only.

## 4. Send the first request

```bash
curl -fS -X POST http://127.0.0.1:8080/v1/recognize \
  -F 'image=@bottle.jpg;type=image/jpeg'
```

Upload image bytes as multipart field `image` (JPEG, PNG, or WebP; max 20 MiB and 32 megapixels). The response contains a suggested `slug`, catalog `card`, up to five `top5` alternatives, raw scores, and `timings_ms`. When `status` is `similar_candidate`, display the card as an analogue, not an exact identification. `confidence` is always `null` and scores are not calibrated probabilities. For the minimal organizer response, use `/v1/eval/predict`, which returns only `{"slug":"..."}`.

## Start the protected Tuna endpoint

Keep the ingress in a second persistent terminal:

```bash
tmux new -s wine-api-tuna
export TUNA_KEY_AUTH="${WINE_ML_API_TOKEN:?Load the Tuna token from your secret manager first}"
while true; do
  .tools/tuna --config "$WINE_APP_ROOT/.private/tuna-cli.yml" http \
    http://127.0.0.1:8080 --subdomain=akcizny-sbor \
    --https-redirect --rate-limit=2
  sleep 2
done
```

Detach with `Ctrl-B`, then `D`. The public URL is `https://akcizny-sbor.ru.tuna.am`; clients send the same externally managed token in `X-Token`. Keep the token outside Git.

## Run the organizer evaluation on the GPU host

The organizer's script needs `bash`, `curl`, `jq`, and `awk`. Run it next to the GPU service so uploading the control photos does not consume the script's fixed 10-second request timeout:

```bash
export PATH="$WINE_APP_ROOT/.tools:$PATH"
bash /path/to/eval/participant_test.sh \
  --images-dir /path/to/eval/queries \
  --manifest /path/to/eval/queries.tsv \
  --endpoint http://127.0.0.1:8080/v1/eval/predict \
  --output /path/to/eval/predictions.jsonl
```

Install `jq` on the GPU host if missing. The verified standalone Linux AMD64 binary can be placed at `$WINE_APP_ROOT/.tools/jq` from the [official jq 1.8.2 release](https://github.com/jqlang/jq/releases/tag/jq-1.8.2); its SHA-256 is `b1c22172dd303f3be49e935aa56aa48a8b7a46e0bc838b4997d3bb451495870f`. Use a new output path for each run because the script refuses to overwrite an existing predictions file.

## Access from a developer laptop

For direct local development without Tuna, open an SSH tunnel to the service's loopback port:

```bash
ssh -N -L 18080:127.0.0.1:8080 SSH_ALIAS
```

In another terminal, test with:

```bash
curl -f http://127.0.0.1:18080/ready
curl -f -F 'image=@bottle.jpg;type=image/jpeg' \
  http://127.0.0.1:18080/v1/recognize
```

The organizer's script can also run on the laptop with `bash`, `curl`, `jq`, and `awk` installed. The script has no `X-Token` option, so use this tunnel rather than the protected public Tuna URL; transfer delays will count toward its 10-second timeout:

```bash
bash /path/to/eval/participant_test.sh \
  --images-dir /path/to/eval/queries \
  --manifest /path/to/eval/queries.tsv \
  --endpoint http://127.0.0.1:18080/v1/eval/predict \
  --output /path/to/predictions.jsonl
```

The eval endpoint returns exactly one non-empty `slug` per image. The full `/v1/recognize` response tells the application whether that slug is a regular candidate or a `similar_candidate` analogue.

For external backends, use `https://akcizny-sbor.ru.tuna.am` and send the token from your secret manager in `X-Token`. Tuna enforces HTTPS and a 2 requests/second rate limit. Never commit the token; rotate the previously committed value before production use.

A backend on the same private network can call `http://PRIVATE_GPU_HOST:8080` directly. The API itself has no authentication or CORS: use the protected Tuna URL for external server-to-server access, and never expose port 8080 directly to the public internet.

## Self-contained clone with weights

If FS2 is unavailable, clone normally (without `GIT_LFS_SKIP_SMUDGE=1`) or run `git lfs pull`, then omit `WINE_MODELS_ROOT` so models come from the repository checkout:

```bash
unset WINE_MODELS_ROOT
git lfs pull
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

This downloads about 5.8 GB of model LFS objects. The setup reassembles Qwen's verified parts locally. For the full backend request/response schema and error handling, see [`api-integration.md`](api-integration.md).
