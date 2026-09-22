import io
import json

import numpy as np
import pytest
from PIL import Image

from wine_api.preprocess import InvalidImage, decode_image, retrieval_views
from wine_api.retrieval import CatalogIndex, is_index_eligible


def index_metadata(dimensions=2):
    return {
        "schema_version": 1,
        "embedding_dimensions": dimensions,
        "catalog_version": "test-catalog",
        "model_source": "test-model",
        "model_revision": "test-revision",
        "gallery_policy": "confirmed",
    }


def test_decode_uses_content_and_composites_alpha():
    source = Image.new("RGBA", (40, 60), (255, 0, 0, 128))
    payload = io.BytesIO()
    source.save(payload, format="WEBP")
    image = decode_image(payload.getvalue())
    assert image.mode == "RGB" and image.size == (40, 60)
    assert image.getpixel((0, 0))[0] == 255 and image.getpixel((0, 0))[1] > 100
    assert [name for name, _ in retrieval_views(image)] == ["whole", "center70"]


def test_decode_rejects_non_image():
    with pytest.raises(InvalidImage):
        decode_image(b"not an image")


def test_decode_rejects_oversized_dimensions():
    payload = io.BytesIO()
    Image.new("1", (6000, 6000)).save(payload, format="PNG")
    with pytest.raises(InvalidImage, match="pixels"):
        decode_image(payload.getvalue())


def test_only_explicitly_confirmed_assets_are_eligible(tmp_path):
    image = tmp_path / "asset.webp"
    Image.new("RGB", (8, 8)).save(image)
    record = {
        "wine_id": "wine-a",
        "image": {"path": str(image)},
        "join": {"confirmed": True},
        "status": {"catalog_image": "ready", "training_eligible": True},
    }
    assert is_index_eligible(record, tmp_path)
    record["join"]["confirmed"] = False
    assert not is_index_eligible(record, tmp_path)


def test_provisional_policy_is_explicit_and_excludes_ambiguity(tmp_path):
    image = tmp_path / "asset.webp"
    Image.new("RGB", (8, 8)).save(image)
    record = {
        "wine_id": "wine-a",
        "image": {"path": str(image)},
        "join": {"confirmed": False, "confidence": "candidate"},
        "status": {"catalog_image": "candidate", "training_eligible": False},
    }
    assert not is_index_eligible(record, tmp_path)
    assert is_index_eligible(record, tmp_path, allow_provisional=True)
    record["status"]["catalog_image"] = "ambiguous"
    assert not is_index_eligible(record, tmp_path, allow_provisional=True)


def test_exact_search_deduplicates_slug():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = CatalogIndex(
        embeddings,
        [
            {"slug": "a", "view": "whole", "card": {"slug": "a"}, "provenance": {}},
            {"slug": "a", "view": "center70", "card": {"slug": "a"}, "provenance": {}},
            {"slug": "b", "view": "whole", "card": {"slug": "b"}, "provenance": {}},
        ],
        index_metadata(),
    )
    results = index.search(np.array([[1.0, 0.0]], dtype=np.float32), limit=5)
    assert [item["slug"] for item in results] == ["a", "b"]
    assert json.dumps(results)


def test_index_rejects_non_normalized_embeddings():
    with pytest.raises(ValueError, match="L2-normalized"):
        CatalogIndex(
            np.array([[2.0, 0.0]], dtype=np.float32),
            [{"slug": "a", "view": "whole", "card": {"slug": "a"}, "provenance": {}}],
            index_metadata(),
        )


def test_search_rejects_non_finite_query():
    index = CatalogIndex(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [{"slug": "a", "view": "whole", "card": {"slug": "a"}, "provenance": {}}],
        index_metadata(),
    )
    with pytest.raises(ValueError, match="non-finite"):
        index.search(np.array([[np.nan, 0.0]], dtype=np.float32))
