import json
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from wine_api.api import Predictor, app, predictor


def test_eval_contract_is_flat_multipart(monkeypatch):
    monkeypatch.setattr(predictor, "predict", lambda _: {"slug": "catalog-slug"})
    response = TestClient(app).post(
        "/v1/eval/predict",
        files={"image": ("wrong.jpg", b"content bytes are handled by the predictor", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {"slug": "catalog-slug"}


def test_predictor_not_ready_for_unversioned_index(tmp_path, monkeypatch):
    path = tmp_path / "bad-index.npz"
    np.savez(
        path,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        entries_json=np.array(json.dumps([{"slug": "a", "card": {"slug": "a"}}])),
        metadata_json=np.array(json.dumps({"schema_version": 1, "embedding_dimensions": 2})),
    )
    monkeypatch.setenv("WINE_INDEX", str(path))
    unready = Predictor()
    unready.load()
    assert not unready.ready
    assert "version metadata" in unready.error


def test_predictor_uses_qwen_margin_when_ocr_candidate_has_no_visual_score():
    class Encoder:
        def embed(self, _):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class Index:
        metadata = {"gallery_policy": "provisional"}

        def search(self, _, limit):
            assert limit == 6
            return [
                {"slug": str(i), "score": None if i == 5 else float(i), "card": {"name": str(i)}, "provenance": {}}
                for i in range(6)
            ]

    class Reranker:
        config = {"candidate_count": 6, "document_text_fields": ["name"], "model_id": "qwen"}
        config_sha256 = "config"

        @staticmethod
        def query_view(image):
            return image

        @staticmethod
        def score(_, text, __):
            return float(text.split()[-1])

    runtime = Predictor()
    runtime.index = Index()
    runtime.encoder = Encoder()
    runtime.reranker = Reranker()
    runtime.manifest = {str(i): {"wine_id": str(i), "image": {"path": "/unused"}} for i in range(6)}
    runtime.manifest_root = None
    image = Image.new("RGB", (8, 8), "white")
    payload = BytesIO()
    image.save(payload, format="PNG")

    result = runtime.predict(payload.getvalue())

    assert result["slug"] == "5"
    assert len(result["top5"]) == 5
    assert result["raw_scores"]["visual_cosine"] is None
    assert result["margin"] == 1.0
