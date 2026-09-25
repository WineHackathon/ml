from __future__ import annotations

import os
import hashlib
import json
import queue
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from . import __version__
from .preprocess import InvalidImage, decode_image, retrieval_views
from .retrieval import CatalogIndex, DEFAULT_MODEL, DEFAULT_REVISION, ImageEncoder, load_manifest, manifest_image_path, manifest_slug

OCR_STARTUP_TIMEOUT_SECONDS = 120
OCR_REQUEST_TIMEOUT_SECONDS = 9
# An in-flight model call cannot be interrupted; the hard 10s service target relies on warmup keeping each call within budget.


class Predictor:
    def __init__(self) -> None:
        self.index_path = Path(os.getenv("WINE_INDEX", "artifacts/catalog_v2/catalog_index.npz"))
        self.index: CatalogIndex | None = None
        self.encoder: ImageEncoder | None = None
        self.reranker = None
        self.manifest: dict[str, dict[str, Any]] = {}
        self.manifest_root: Path | None = None
        self.gallery_by_slug: dict[str, dict[str, Any]] = {}
        self.fusion = None
        self.ocr_python: Path | None = None
        self.ocr_script: Path | None = None
        self.ocr_cache: Path | None = None
        self.ocr_worker = None
        self.ocr_worker_enabled = False
        self.ocr_worker_lock = Lock()
        self.error: str | None = None

    @staticmethod
    def _stop_process(process) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _stop_ocr_worker(self) -> None:
        worker, self.ocr_worker = self.ocr_worker, None
        if worker:
            self._stop_process(worker)

    def _read_ocr_worker_line(self, timeout: float) -> str:
        worker = self.ocr_worker
        if not worker:
            raise RuntimeError("OCR worker is unavailable")
        result = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put((worker.stdout.readline(), None))
            except Exception as exc:
                result.put(("", exc))

        Thread(target=read, daemon=True).start()
        try:
            line, error = result.get(timeout=timeout)
        except queue.Empty:
            message = f"OCR worker timed out after {timeout:g}s"
            self.error = f"RuntimeError: {message}"
            self._stop_ocr_worker()
            raise RuntimeError(message) from None
        if error or not line:
            self.error = "RuntimeError: OCR worker closed without a response"
            self._stop_ocr_worker()
            raise RuntimeError("OCR worker closed without a response") from error
        return line

    def load(self) -> None:
        self.ocr_worker_enabled = False
        try:
            self.index = CatalogIndex.load(self.index_path)
            metadata = self.index.metadata
            model = os.getenv("WINE_MODEL", metadata.get("model_source") or DEFAULT_MODEL)
            revision = os.getenv("WINE_MODEL_REVISION", metadata.get("model_revision") or DEFAULT_REVISION)
            indexed_revision = metadata.get("model_revision")
            if indexed_revision and revision != indexed_revision:
                raise ValueError("runtime model revision differs from the index")
            if model != metadata.get("model_source"):
                local_snapshot = Path(model)
                if not local_snapshot.is_dir() or local_snapshot.name != indexed_revision:
                    raise ValueError("WINE_MODEL override is not the indexed revision snapshot")
            self.encoder = ImageEncoder(
                model,
                revision,
                os.getenv("WINE_DEVICE", "auto"),
                os.getenv("WINE_LOCAL_ONLY") == "1",
            )
            if self.encoder.output_dimensions != self.index.embeddings.shape[1]:
                raise ValueError("runtime model dimensions differ from the index")
            reranker_mode = os.getenv("WINE_RERANKER", "qwen")
            if reranker_mode == "qwen":
                manifest_path = Path(os.getenv("WINE_MANIFEST", "data/catalog_manifest.jsonl"))
                config_path = Path(os.getenv("WINE_RERANKER_CONFIG", "artifacts/qwen_reranker/config.json"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if hashlib.sha256(self.index_path.read_bytes()).hexdigest() != config["base_index_sha256"]:
                    raise ValueError("reranker config index hash mismatch")
                if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config["manifest_sha256"]:
                    raise ValueError("reranker config manifest hash mismatch")
                self.manifest = {manifest_slug(row): row for row in load_manifest(manifest_path)}
                self.manifest_root = next(parent for parent in manifest_path.resolve().parents if (parent / "pyproject.toml").is_file())
                required_slugs = sorted({entry["slug"] for entry in self.index.entries})
                missing_references = []
                reference_paths = set()
                for slug in required_slugs:
                    record = self.manifest.get(slug)
                    path = manifest_image_path(record, self.manifest_root) if record else None
                    if not path or not path.is_file():
                        missing_references.append(slug)
                    else:
                        reference_paths.add(path.resolve())
                if missing_references:
                    sample = ", ".join(missing_references[:3]) + (", ..." if len(missing_references) > 3 else "")
                    raise ValueError(f"missing {len(missing_references)} indexed reranker references: {sample}")
                from .qwen_reranker import QwenReranker

                self.reranker = QwenReranker(
                    Path(os.getenv("WINE_RERANKER_MODEL", "models/Qwen3-VL-Reranker-2B")), config_path
                )
                self.reranker.preload_references(sorted(reference_paths))
                self.gallery_by_slug = {
                    entry["slug"]: {"slug": entry["slug"], "score": None, "card": entry["card"], "provenance": entry["provenance"]}
                    for entry in self.index.entries
                }
                if config.get("candidate_policy") == "frozen_siglip_top6_ocr_entity_replacement":
                    from .candidate_fusion import CONFIG, fuse, load_catalog

                    fusion_sha = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
                    if fusion_sha != config["candidate_fusion_config_sha256"]:
                        raise ValueError("candidate fusion config hash mismatch")
                    catalog, producer_df, assets = load_catalog(manifest_path)
                    self.fusion = (fuse, catalog, producer_df, assets)
                    self.ocr_python = Path(os.getenv("WINE_OCR_PYTHON", ".venv-ocr/bin/python"))
                    self.ocr_script = Path(os.environ.get("WINE_OCR_SCRIPT", "scripts/ocr_rerank.py")).resolve()
                    self.ocr_cache = Path(os.environ.get("WINE_OCR_CACHE", "artifacts/live_ocr_cache.json"))
                    if os.getenv("WINE_OCR_WORKER") == "1":
                        self.ocr_worker_enabled = True
                        self.ocr_worker = subprocess.Popen(
                            [str(self.ocr_python), str(self.ocr_script.with_name("ocr_worker.py"))],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                        )
                        ready = json.loads(self._read_ocr_worker_line(OCR_STARTUP_TIMEOUT_SECONDS))
                        worker_config_sha = hashlib.sha256(json.dumps(ready["config"], sort_keys=True).encode()).hexdigest()
                        if not ready.get("ready") or worker_config_sha != config["ocr_config_sha256"]:
                            raise ValueError("OCR worker readiness/config mismatch")
                if config.get("startup_warmup"):
                    self._warmup()
            elif reranker_mode != "off":
                raise ValueError("WINE_RERANKER must be qwen or off")
            self.error = None
        except Exception as exc:
            self.close()
            self.index = None
            self.encoder = None
            self.reranker = None
            self.fusion = None
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        if self.index is None or self.encoder is None:
            return False
        if self.ocr_worker_enabled and (not self.ocr_worker or self.ocr_worker.poll() is not None):
            self.error = self.error or "RuntimeError: OCR worker is unavailable"
            return False
        return True

    def close(self) -> None:
        self._stop_ocr_worker()
        if self.reranker and hasattr(self.reranker, "close"):
            self.reranker.close()

    def _warmup(self) -> None:
        image = Image.new("RGB", (256, 256), "white")
        views = retrieval_views(image)
        self.encoder.embed([view for _, view in views])
        for _, view in views:
            if view is not image:
                view.close()
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as handle:
            path = Path(handle.name)
        try:
            image.save(path, format="WEBP")
            if self.ocr_worker:
                self.ocr_worker.stdin.write(json.dumps({"path": str(path)}) + "\n")
                self.ocr_worker.stdin.flush()
                response = json.loads(self._read_ocr_worker_line(OCR_REQUEST_TIMEOUT_SECONDS))
                if "error" in response:
                    raise RuntimeError(f"OCR warmup failed: {response['error']}")
            candidates = [self.gallery_by_slug[slug] for slug in sorted(self.gallery_by_slug)[:6]]
            documents = [
                (candidate["card"], manifest_image_path(self.manifest[candidate["slug"]], self.manifest_root))
                for candidate in candidates
            ]
            self.reranker.rank(image, documents, path)
        finally:
            image.close()
            path.unlink(missing_ok=True)

    def predict(self, data: bytes) -> dict[str, Any]:
        cleanups: list[Callable[[], None]] = []
        try:
            return self._predict(data, cleanups)
        finally:
            for cleanup in reversed(cleanups):
                cleanup()

    def _predict(self, data: bytes, cleanups: list[Callable[[], None]]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(self.error or "model/index not loaded")
        started = time.perf_counter()
        image = decode_image(data)
        decoded = time.perf_counter()
        views = retrieval_views(image)
        embeddings = self.encoder.embed([view for _, view in views])
        embedded = time.perf_counter()
        limit = 20 if self.fusion else self.reranker.config["candidate_count"] if self.reranker else 5
        visual_candidates = self.index.search(embeddings, limit=limit)
        searched = time.perf_counter()
        if not visual_candidates:
            raise RuntimeError("empty retrieval result")
        ocr_process = None
        temporary_path = None
        degraded_reason = None
        ocr_seconds = None
        qwen_seconds = 0.0
        request_deadline = started + OCR_REQUEST_TIMEOUT_SECONDS
        if self.fusion:
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as handle:
                handle.write(data)
                temporary_path = Path(handle.name)
                cleanups.append(lambda path=temporary_path: path.unlink(missing_ok=True))
            use_ocr_worker = self.ocr_worker_enabled or self.ocr_worker is not None
            if not use_ocr_worker:
                ocr_process = subprocess.Popen(
                    [str(self.ocr_python), str(self.ocr_script), "--cache", str(self.ocr_cache), str(temporary_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                cleanups.append(lambda process=ocr_process: self._stop_process(process))
            candidates = visual_candidates[:5]
        else:
            candidates = visual_candidates
        if self.reranker:
            fields = self.reranker.config["document_text_fields"]
            query_image = self.reranker.query_view(image)
            listwise = hasattr(self.reranker, "rank")
            worker_response_line = None
            with self.ocr_worker_lock if self.fusion and use_ocr_worker else nullcontext():
                if self.fusion and use_ocr_worker:
                    if not self.ocr_worker:
                        raise RuntimeError("OCR worker is unavailable")
                    if time.perf_counter() >= request_deadline:
                        raise RuntimeError(f"OCR request exceeded {OCR_REQUEST_TIMEOUT_SECONDS}s deadline")
                    try:
                        self.ocr_worker.stdin.write(json.dumps({"path": str(temporary_path)}) + "\n")
                        self.ocr_worker.stdin.flush()
                    except Exception as exc:
                        self.error = "RuntimeError: OCR worker request failed"
                        self._stop_ocr_worker()
                        raise RuntimeError("OCR worker request failed") from exc
                score_failed = False
                try:
                    if not listwise:
                        for candidate in candidates:
                            if time.perf_counter() >= request_deadline:
                                raise RuntimeError(f"request exceeded {OCR_REQUEST_TIMEOUT_SECONDS}s deadline")
                            record = self.manifest[candidate["slug"]]
                            text = "\n".join(f"{field}: {candidate['card'][field]}" for field in fields if candidate["card"].get(field))
                            score_started = time.perf_counter()
                            candidate["reranker_score"] = self.reranker.score(query_image, text, manifest_image_path(record, self.manifest_root))
                            qwen_seconds += time.perf_counter() - score_started
                except BaseException:
                    score_failed = True
                    raise
                finally:
                    if self.fusion and use_ocr_worker:
                        try:
                            worker_response_line = self._read_ocr_worker_line(max(0, request_deadline - time.perf_counter()))
                        except Exception:
                            if not score_failed:
                                raise
            if self.fusion:
                if use_ocr_worker:
                    worker_response = json.loads(worker_response_line)
                    if "error" in worker_response:
                        raise RuntimeError(f"OCR failed: {worker_response['error']}")
                    ocr = worker_response["result"]
                    ocr_seconds = ocr["seconds"]
                else:
                    timeout = max(0, request_deadline - time.perf_counter())
                    try:
                        stdout, stderr = ocr_process.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f"OCR process timed out after {OCR_REQUEST_TIMEOUT_SECONDS}s") from None
                    if ocr_process.returncode:
                        raise RuntimeError(f"OCR failed: {stderr.strip()}")
                    ocr_payload = json.loads(stdout)
                    ocr = ocr_payload["results"][str(temporary_path)]
                    ocr_seconds = ocr_payload["timings"]["batch_wall_seconds"]
                ocr_config_sha = hashlib.sha256(json.dumps(ocr["config"], sort_keys=True).encode()).hexdigest()
                if ocr_config_sha != self.reranker.config["ocr_config_sha256"]:
                    raise RuntimeError("OCR config hash mismatch")
                fuse, catalog, producer_df, assets = self.fusion
                pool = fuse(
                    [{"slug": item["slug"], "score": item["score"]} for item in visual_candidates],
                    ocr["text"],
                    catalog,
                    producer_df,
                    assets,
                )
                scored = {item["slug"]: item for item in candidates}
                candidates = []
                for evidence in pool:
                    candidate = dict(scored.get(evidence["slug"]) or self.gallery_by_slug[evidence["slug"]])
                    candidate.update({
                        "ocr_entity_score": evidence["ocr_entity_score"],
                        "ocr_year_match": evidence["ocr_year_match"],
                        "ocr_year_mismatch": evidence["ocr_year_mismatch"],
                    })
                    if not listwise and "reranker_score" not in candidate:
                        if time.perf_counter() >= request_deadline:
                            raise RuntimeError(f"request exceeded {OCR_REQUEST_TIMEOUT_SECONDS}s deadline")
                        record = self.manifest[candidate["slug"]]
                        text = "\n".join(f"{field}: {candidate['card'][field]}" for field in fields if candidate["card"].get(field))
                        score_started = time.perf_counter()
                        candidate["reranker_score"] = self.reranker.score(query_image, text, manifest_image_path(record, self.manifest_root))
                        qwen_seconds += time.perf_counter() - score_started
                    candidates.append(candidate)
            if listwise:
                documents = [
                    (candidate["card"], manifest_image_path(self.manifest[candidate["slug"]], self.manifest_root))
                    for candidate in candidates
                ]
                try:
                    if time.perf_counter() >= request_deadline:
                        raise RuntimeError(f"request exceeded {OCR_REQUEST_TIMEOUT_SECONDS}s deadline")
                    score_started = time.perf_counter()
                    scores = self.reranker.rank(query_image, documents, temporary_path)
                except TimeoutError as exc:
                    degraded_reason = str(exc)
                else:
                    qwen_seconds += time.perf_counter() - score_started
                    for candidate, score in zip(candidates, scores):
                        candidate["reranker_score"] = score
            if query_image is not image:
                query_image.close()
            if not degraded_reason:
                candidates.sort(
                    key=(
                        (lambda item: (-item["reranker_score"], item["slug"]))
                        if listwise
                        else (lambda item: (-item["reranker_score"], -item.get("ocr_entity_score", 0.0), item["slug"]))
                    )
                )
        top5 = candidates[:5]
        finished = time.perf_counter()
        margin = None
        if len(top5) > 1:
            margin = (
                top5[0]["reranker_score"] - top5[1]["reranker_score"]
                if self.reranker and not degraded_reason
                else top5[0]["score"] - top5[1]["score"]
            )
        gallery_policy = self.index.metadata.get("gallery_policy", "confirmed")
        return {
            "status": "degraded_candidate" if degraded_reason else "provisional_candidate" if gallery_policy == "provisional" else "candidate",
            "degraded": bool(degraded_reason),
            "degraded_reason": degraded_reason,
            "slug": top5[0]["slug"],
            "card": top5[0]["card"],
            "confidence": None,
            "uncalibrated": True,
            "raw_scores": {
                "visual_cosine": top5[0]["score"],
                "reranker": top5[0].get("reranker_score"),
            },
            "margin": margin,
            "identity": top5[0]["provenance"],
            "top5": top5,
            "timings_ms": {
                "decode": round((decoded - started) * 1000, 2),
                "embedding": round((embedded - decoded) * 1000, 2),
                "search": round((searched - embedded) * 1000, 2),
                "ocr": round(ocr_seconds * 1000, 2) if ocr_seconds is not None else None,
                "qwen": round(qwen_seconds * 1000, 2),
                "rerank": round((finished - searched) * 1000, 2),
                "total": round((finished - started) * 1000, 2),
            },
            "catalog_version": self.index.metadata.get("catalog_version"),
            "gallery_policy": gallery_policy,
            "catalog_coverage": {
                "indexed_slugs": self.index.metadata.get("indexed_slugs"),
                "manifest_records": self.index.metadata.get("manifest_records"),
                "ratio": self.index.metadata.get("coverage_ratio"),
                "identity_status": self.index.metadata.get("indexed_identity_status"),
            },
            "model_version": {
                "source": self.index.metadata.get("model_source"),
                "revision": self.index.metadata.get("model_revision"),
                "reranker": self.reranker.config["model_id"] if self.reranker else None,
                "reranker_config_sha256": self.reranker.config_sha256 if self.reranker else None,
            },
        }


predictor = Predictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    try:
        yield
    finally:
        predictor.close()


app = FastAPI(title="Wine ML API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> JSONResponse:
    payload = {"ready": predictor.ready, "error": predictor.error}
    return JSONResponse(payload, status_code=200 if predictor.ready else 503)


async def _read_upload(image: UploadFile) -> bytes:
    from .preprocess import MAX_UPLOAD_BYTES

    return await image.read(MAX_UPLOAD_BYTES + 1)


@app.post("/v1/recognize")
async def recognize(image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        return predictor.predict(await _read_upload(image))
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> dict[str, str]:
    result = await recognize(image)
    return {"slug": result["slug"]}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "wine_api.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        workers=1,
    )
