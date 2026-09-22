#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/catalog_manifest.jsonl"
INDEX = ROOT / "artifacts/catalog_v2/catalog_index.npz"
CONFIG = ROOT / "artifacts/qwen_reranker/config.json"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    with np.load(INDEX, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        embeddings = data["embeddings"]
    checks = {
        "manifest": hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == config["manifest_sha256"] == metadata["manifest_sha256"],
        "index": hashlib.sha256(INDEX.read_bytes()).hexdigest() == config["base_index_sha256"],
        "embeddings": hashlib.sha256(embeddings.tobytes()).hexdigest() == metadata["embeddings_sha256"],
        "coverage": metadata["indexed_slugs"] == 2088 and metadata["manifest_records"] == 2103,
    }
    print(json.dumps(checks, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

