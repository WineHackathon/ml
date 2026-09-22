#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebase frozen index metadata onto a path-sanitized manifest")
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        entries_json = str(data["entries_json"].item())
        metadata = json.loads(str(data["metadata_json"].item()))
    manifest_sha = sha(args.manifest)
    metadata.update({
        "source_catalog_version": metadata["catalog_version"],
        "source_index_sha256": sha(args.source),
        "catalog_version": manifest_sha[:16],
        "manifest_sha256": manifest_sha,
        "embeddings_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "entries_sha256": hashlib.sha256(entries_json.encode()).hexdigest(),
        "path_rebase_only": True,
    })
    np.savez(
        args.output,
        embeddings=embeddings,
        entries_json=np.array(entries_json),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )


if __name__ == "__main__":
    main()

