import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "data/catalog_manifest.jsonl"
INDEX = ROOT / "artifacts/catalog_v2/catalog_index.npz"
CONFIG = ROOT / "artifacts/qwen_reranker/config.json"
H100_CONFIG = ROOT / "artifacts/qwen_reranker/config-h100.json"
SPEC = importlib.util.spec_from_file_location("provision_assets", ROOT / "scripts/provision_assets.py")
provision_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provision_assets)


def rows():
    return [json.loads(line) for line in MANIFEST.read_text().splitlines()]


def test_manifest_has_portable_paths_and_expected_coverage():
    records = rows()
    assert len(records) == 2103
    assert all(not (row.get("image") or {}).get("path", "").startswith("/") for row in records)
    assert all("/Users/" not in json.dumps(row) for row in records)


def test_index_metadata_matches_sanitized_manifest():
    with np.load(INDEX, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert metadata["manifest_sha256"] == manifest_sha
    assert metadata["catalog_version"] == manifest_sha[:16]
    assert metadata["path_rebase_only"] is False


def test_index_embeddings_are_frozen_and_cover_2087_slugs():
    with np.load(INDEX, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert hashlib.sha256(data["embeddings"].tobytes()).hexdigest() == metadata["embeddings_sha256"]
        assert metadata["indexed_slugs"] == 2087
        assert metadata["manifest_records"] == 2103


def test_reranker_config_locks_checked_in_artifacts():
    config = json.loads(CONFIG.read_text())
    assert config["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert config["base_index_sha256"] == hashlib.sha256(INDEX.read_bytes()).hexdigest()

    h100_config = json.loads(H100_CONFIG.read_text())
    assert h100_config["dtype"] == "float32"
    assert h100_config["manifest_sha256"] == config["manifest_sha256"]
    assert h100_config["base_index_sha256"] == config["base_index_sha256"]


def test_local_asset_import_is_hash_addressed(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    payload = b"not decoded here; runtime validates images"
    sha = hashlib.sha256(payload).hexdigest()
    (source / "organizer-name.webp").write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"wine_id": "x", "image": {"sha256": sha, "format": "webp"}}) + "\n")
    copied, missing = provision_assets.provision(manifest, [source], False, destination)
    assert (copied, missing) == (1, 0)
    assert (destination / f"{sha}.webp").read_bytes() == payload
