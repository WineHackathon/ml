#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "models"
    models = json.loads((project / "models.lock.json").read_text())
    for model in models:
        directory = root / model["directory"]
        snapshot_download(model["repo"], revision=model["revision"], local_dir=directory)
        actual = hashlib.sha256((directory / model["weight"]).read_bytes()).hexdigest()
        if actual != model["sha256"]:
            raise RuntimeError(f"model hash mismatch: {model['repo']}")


if __name__ == "__main__":
    main()
