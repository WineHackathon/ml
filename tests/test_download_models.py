import hashlib

import pytest

from scripts.download_models import materialize


def test_materializes_verified_split_weight_offline(tmp_path):
    payload = b"model weights" * 100
    directory = tmp_path / "model"
    directory.mkdir()
    parts = [payload[:333], payload[333:]]
    part_specs = []
    for name, contents in zip(("weights.aa", "weights.ab"), parts):
        (directory / name).write_bytes(contents)
        part_specs.append({"file": name, "sha256": hashlib.sha256(contents).hexdigest()})
    model = {
        "repo": "org/model",
        "directory": "model",
        "weight": "weights.bin",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parts": part_specs,
    }

    materialize(model, tmp_path, offline=True)

    assert (directory / "weights.bin").read_bytes() == payload
    assert not (directory / "weights.bin.assembling").exists()


def test_rejects_corrupt_offline_split(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "weights.aa").write_bytes(b"corrupt")
    model = {
        "repo": "org/model",
        "directory": "model",
        "weight": "weights.bin",
        "sha256": hashlib.sha256(b"expected").hexdigest(),
        "parts": [{"file": "weights.aa", "sha256": hashlib.sha256(b"expected").hexdigest()}],
    }

    with pytest.raises(RuntimeError, match="part hash mismatch"):
        materialize(model, tmp_path, offline=True)

    assert not (directory / "weights.bin").exists()
