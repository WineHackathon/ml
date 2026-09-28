#!/usr/bin/env python3
import hashlib
import json
import argparse
import os
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(model: dict, root: Path, offline: bool) -> None:
    directory = root / model["directory"]
    weight = directory / model["weight"]
    if weight.is_file() and sha256(weight) == model["sha256"]:
        return

    parts = model.get("parts", [])
    if parts and all((directory / part["file"]).is_file() for part in parts):
        temporary = weight.with_name(weight.name + ".assembling")
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as output:
                for part in parts:
                    source = directory / part["file"]
                    if sha256(source) != part["sha256"]:
                        raise RuntimeError(f"model part hash mismatch: {source}")
                    with source.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                            output.write(chunk)
            if digest.hexdigest() != model["sha256"]:
                raise RuntimeError(f"assembled model hash mismatch: {model['repo']}")
            os.replace(temporary, weight)
            return
        finally:
            temporary.unlink(missing_ok=True)

    if offline:
        raise FileNotFoundError(f"missing verified weight or split parts: {weight}")
    from huggingface_hub import snapshot_download

    snapshot_download(model["repo"], revision=model["revision"], local_dir=directory)
    if not weight.is_file() or sha256(weight) != model["sha256"]:
        raise RuntimeError(f"model hash mismatch: {model['repo']}")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project / "models")
    parser.add_argument("--offline", action="store_true", help="only use bundled model files")
    args = parser.parse_args()
    models = json.loads((project / "models.lock.json").read_text())
    for model in models:
        materialize(model, args.root, args.offline)


if __name__ == "__main__":
    main()
