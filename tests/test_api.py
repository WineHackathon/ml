import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from io import BytesIO
from threading import Event, Thread

import numpy as np
import pytest
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


def test_predictor_rejects_missing_indexed_reranker_reference(tmp_path, monkeypatch):
    index_path = tmp_path / "index.npz"
    index_path.write_bytes(b"index")
    manifest_path = tmp_path / "data" / "manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"wine_id": "missing", "image": {"path": "missing.webp"}}) + "\n")
    (tmp_path / "pyproject.toml").touch()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "base_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }))
    index = SimpleNamespace(
        metadata={"model_source": "model", "model_revision": "revision"},
        embeddings=np.zeros((1, 2)),
        entries=[{"slug": "missing"}],
    )
    monkeypatch.setattr("wine_api.api.CatalogIndex.load", lambda _: index)
    monkeypatch.setattr("wine_api.api.ImageEncoder", lambda *_: SimpleNamespace(output_dimensions=2))
    monkeypatch.setitem(sys.modules, "wine_api.qwen_reranker", SimpleNamespace(QwenReranker=pytest.fail))
    monkeypatch.setenv("WINE_INDEX", str(index_path))
    monkeypatch.setenv("WINE_MODEL", "model")
    monkeypatch.setenv("WINE_MODEL_REVISION", "revision")
    monkeypatch.setenv("WINE_MANIFEST", str(manifest_path))
    monkeypatch.setenv("WINE_RERANKER_CONFIG", str(config_path))

    runtime = Predictor()
    runtime.load()

    assert not runtime.ready
    assert runtime.error == "ValueError: missing 1 indexed reranker references: missing"


def test_predictor_load_retry_resets_disabled_ocr_worker(monkeypatch):
    index = SimpleNamespace(
        metadata={"model_source": "model", "model_revision": "revision"},
        embeddings=np.zeros((1, 2)),
    )
    monkeypatch.setattr("wine_api.api.CatalogIndex.load", lambda _: index)
    monkeypatch.setattr("wine_api.api.ImageEncoder", lambda *_: SimpleNamespace(output_dimensions=2))
    monkeypatch.setenv("WINE_MODEL", "model")
    monkeypatch.setenv("WINE_MODEL_REVISION", "revision")
    monkeypatch.setenv("WINE_RERANKER", "off")
    monkeypatch.setenv("WINE_OCR_WORKER", "0")
    runtime = Predictor()
    runtime.ocr_worker_enabled = True

    runtime.load()

    assert runtime.ready
    assert not runtime.ocr_worker_enabled


def test_predictor_uses_qwen_margin_when_ocr_candidate_has_no_visual_score(monkeypatch):
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

    monkeypatch.setattr("wine_api.api.OCR_REQUEST_TIMEOUT_SECONDS", 0)
    runtime.reranker.score = lambda *_: pytest.fail("expired request started Qwen")
    with pytest.raises(RuntimeError, match="request exceeded 0s deadline"):
        runtime.predict(payload.getvalue())


def test_persistent_ocr_starts_before_qwen_scoring():
    events = []
    ocr_config = {"engine": "fake"}

    class Encoder:
        def embed(self, _):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class Index:
        metadata = {}

        def search(self, _, limit):
            return [
                {"slug": str(i), "score": float(5 - i), "card": {"name": str(i)}, "provenance": {}}
                for i in range(6)
            ]

    class Reranker:
        config = {
            "candidate_count": 6,
            "document_text_fields": ["name"],
            "model_id": "qwen",
            "ocr_config_sha256": hashlib.sha256(json.dumps(ocr_config, sort_keys=True).encode()).hexdigest(),
        }
        config_sha256 = "config"

        @staticmethod
        def query_view(image):
            return image

        @staticmethod
        def score(*_):
            events.append("qwen")
            return 1.0

    class Input:
        def write(self, _):
            events.append("ocr-write")

        def flush(self):
            events.append("ocr-flush")

    class Output:
        @staticmethod
        def readline():
            events.append("ocr-read")
            return json.dumps({"result": {"config": ocr_config, "text": "", "seconds": 0.1}})

    def fuse(visual, *_):
        return [
            {"slug": item["slug"], "ocr_entity_score": 0.0, "ocr_year_match": False, "ocr_year_mismatch": False}
            for item in visual[:5]
        ]

    runtime = Predictor()
    runtime.index = Index()
    runtime.encoder = Encoder()
    runtime.reranker = Reranker()
    runtime.fusion = (fuse, {}, {}, {})
    runtime.ocr_worker = type("Worker", (), {"stdin": Input(), "stdout": Output()})()
    runtime.manifest = {str(i): {"image": {"path": "/unused"}} for i in range(6)}
    runtime.manifest_root = None
    image = Image.new("RGB", (8, 8), "white")
    payload = BytesIO()
    image.save(payload, format="PNG")

    runtime.predict(payload.getvalue())

    assert events[:3] == ["ocr-write", "ocr-flush", "qwen"]
    assert events[-1] == "ocr-read"


def test_persistent_ocr_timeout_kills_and_invalidates_worker(monkeypatch):
    stopped = Event()

    class Output:
        @staticmethod
        def readline():
            stopped.wait()
            return ""

    class Worker:
        stdout = Output()
        terminated = killed = False

        @staticmethod
        def poll():
            return None

        def terminate(self):
            self.terminated = True
            stopped.set()

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            self.killed = True

    runtime = Predictor()
    runtime.index = runtime.encoder = object()
    runtime.ocr_worker_enabled = True
    worker = runtime.ocr_worker = Worker()

    with pytest.raises(RuntimeError, match="timed out"):
        runtime._read_ocr_worker_line(0.01)

    assert runtime.ocr_worker is None
    assert worker.terminated and worker.killed
    monkeypatch.setattr("wine_api.api.predictor", runtime)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert "timed out" in response.json()["error"]


def test_persistent_ocr_eof_invalidates_worker(monkeypatch):
    class Worker:
        stdout = type("Output", (), {"readline": staticmethod(lambda: "")})()

        @staticmethod
        def poll():
            return 1

    runtime = Predictor()
    runtime.index = runtime.encoder = object()
    runtime.ocr_worker_enabled = True
    runtime.ocr_worker = Worker()

    with pytest.raises(RuntimeError, match="closed without a response"):
        runtime._read_ocr_worker_line(1)

    assert runtime.ocr_worker is None
    monkeypatch.setattr("wine_api.api.predictor", runtime)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert "closed without a response" in response.json()["error"]


def test_queued_request_fails_explicitly_after_worker_timeout():
    class Encoder:
        def embed(self, _):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class Index:
        metadata = {}

        def search(self, *_args, **_kwargs):
            return [{"slug": str(i), "score": float(i), "card": {"name": str(i)}, "provenance": {}} for i in range(6)]

    class Reranker:
        config = {"candidate_count": 6, "document_text_fields": ["name"], "model_id": "qwen"}
        config_sha256 = "config"

        @staticmethod
        def query_view(image):
            return image

        @staticmethod
        def score(*_):
            return 1.0

    class Input:
        @staticmethod
        def write(_):
            pass

        @staticmethod
        def flush():
            pass

    runtime = Predictor()
    runtime.index = Index()
    runtime.encoder = Encoder()
    runtime.reranker = Reranker()
    runtime.fusion = (None, None, None, None)
    runtime.ocr_worker_enabled = True
    runtime.ocr_worker = type("Worker", (), {"stdin": Input(), "poll": staticmethod(lambda: None)})()
    runtime.manifest = {str(i): {"image": {"path": "/unused"}} for i in range(6)}
    runtime.manifest_root = None
    read_started, release = Event(), Event()

    def timeout(_):
        read_started.set()
        release.wait()
        runtime.ocr_worker = None
        raise RuntimeError("OCR worker timed out")

    runtime._read_ocr_worker_line = timeout
    image = Image.new("RGB", (8, 8), "white")
    payload = BytesIO()
    image.save(payload, format="PNG")
    errors = []

    def predict():
        try:
            runtime.predict(payload.getvalue())
        except RuntimeError as exc:
            errors.append(str(exc))

    first = Thread(target=predict)
    first.start()
    assert read_started.wait(1)
    second = Thread(target=predict)
    second.start()
    release.set()
    first.join(1)
    second.join(1)

    assert sorted(errors) == ["OCR worker is unavailable", "OCR worker timed out"]
