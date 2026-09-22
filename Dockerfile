FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    WINE_DEVICE=cpu \
    WINE_RERANKER_DEVICE=cpu \
    WINE_OCR_PYTHON=/opt/wine-ocr/bin/python

WORKDIR /app
COPY pyproject.toml requirements.lock requirements-ocr.lock README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && python -m venv /opt/wine-ocr \
    && /opt/wine-ocr/bin/pip install --no-cache-dir -r requirements-ocr.lock

COPY artifacts ./artifacts
COPY data/catalog_manifest.jsonl ./data/catalog_manifest.jsonl
COPY scripts ./scripts
EXPOSE 8080
CMD ["./scripts/run.sh"]

