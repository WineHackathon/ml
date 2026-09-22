#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


MAX_DOWNLOAD = 25 * 1024 * 1024
ALLOWED_PUBLIC_HOSTS = {"api.vino-svoe.ru"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_manifest(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image = row.get("image")
        if image:
            records[image["sha256"]] = row
    return records


def import_local(sources: list[Path], pending: dict[str, dict], destination: Path) -> int:
    copied = 0
    for source in sources:
        for path in source.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            sha = digest(path)
            if sha in pending:
                target = destination / f"{sha}.{pending[sha]['image']['format'].lower()}"
                shutil.copyfile(path, target)
                pending.pop(sha)
                copied += 1
    return copied


def fetch_public(pending: dict[str, dict], destination: Path) -> int:
    copied = 0
    for sha, row in list(pending.items()):
        url = (row.get("provision") or {}).get("official_image_url")
        parsed = urlparse(url or "")
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_PUBLIC_HOSTS:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "wine-ml-provision/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_DOWNLOAD + 1)
        if len(payload) > MAX_DOWNLOAD or hashlib.sha256(payload).hexdigest() != sha:
            raise RuntimeError(f"official asset failed size/hash validation: {row['wine_id']}")
        target = destination / f"{sha}.{row['image']['format'].lower()}"
        target.write_bytes(payload)
        pending.pop(sha)
        copied += 1
    return copied


def provision(manifest: Path, sources: list[Path], public: bool, destination: Path) -> tuple[int, int]:
    destination.mkdir(parents=True, exist_ok=True)
    pending = load_manifest(manifest)
    for sha in list(pending):
        path = destination / f"{sha}.{pending[sha]['image']['format'].lower()}"
        if path.is_file() and digest(path) == sha:
            pending.pop(sha)
    copied = import_local(sources, pending, destination)
    if public:
        copied += fetch_public(pending, destination)
    return copied, len(pending)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Import catalog references by SHA-256; no source images are published")
    parser.add_argument("sources", nargs="*", type=Path, help="extracted organizer catalog directories")
    parser.add_argument("--fetch-public", action="store_true", help="fetch only allowlisted official URLs embedded in the manifest")
    parser.add_argument("--manifest", type=Path, default=root / "data/catalog_manifest.jsonl")
    parser.add_argument("--destination", type=Path, default=root / "data/catalog_images")
    args = parser.parse_args()
    copied, missing = provision(args.manifest, args.sources, args.fetch_public, args.destination)
    print(json.dumps({"copied": copied, "missing": missing, "destination": str(args.destination)}))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

