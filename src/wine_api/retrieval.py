from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .preprocess import MAX_CATALOG_PIXELS, decode_image, retrieval_views

DEFAULT_MODEL = "google/siglip2-base-patch16-384"
DEFAULT_REVISION = "f775b65a79762255128c981547af89addcfe0f88"


def _nested(record: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def manifest_slug(record: dict[str, Any]) -> str:
    return str(record.get("wine_id") or record.get("slug") or "").strip()


def manifest_image_path(record: dict[str, Any], repo_root: Path) -> Path:
    raw = _nested(record, "image", "path") or record.get("image_path") or ""
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def is_index_eligible(record: dict[str, Any], repo_root: Path, allow_provisional: bool = False) -> bool:
    status = _nested(record, "status", "catalog_image")
    eligible = _nested(record, "status", "training_eligible")
    confirmed = _nested(record, "join", "confirmed")
    if status is None:  # Explicit legacy form, never infer trust from a unique filename.
        status = record.get("join_status")
        eligible = record.get("training_eligible")
        confirmed = record.get("join_confirmed")
    confirmed_ready = bool(
        manifest_slug(record)
        and status in {"ready", "confirmed"}
        and eligible is True
        and confirmed is True
        and manifest_image_path(record, repo_root).is_file()
    )
    provisional_ready = bool(
        allow_provisional
        and manifest_slug(record)
        and status == "candidate"
        and _nested(record, "join", "confidence") == "candidate"
        and confirmed is False
        and manifest_image_path(record, repo_root).is_file()
    )
    return confirmed_ready or provisional_ready


def public_card(record: dict[str, Any]) -> dict[str, Any]:
    slug = manifest_slug(record)
    return {
        "slug": slug,
        "name": record.get("canonical_name") or record.get("name"),
        "producer": record.get("producer"),
        "region": record.get("region") or record.get("country_region"),
        "color": record.get("color"),
        "type": record.get("type"),
        "grapes": record.get("grapes"),
        "description": record.get("description"),
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            records.append(record)
    return records


class ImageEncoder:
    def __init__(self, model_source: str, revision: str | None, device: str, local_only: bool = False):
        self.model_source = model_source
        self.revision = revision
        self.device = choose_device(device)
        kwargs = {"revision": revision, "local_files_only": local_only, "trust_remote_code": False}
        self.processor = AutoProcessor.from_pretrained(model_source, **kwargs)
        self.model = AutoModel.from_pretrained(model_source, **kwargs).eval().to(self.device)
        vision_config = getattr(self.model.config, "vision_config", self.model.config)
        self.output_dimensions = int(
            getattr(vision_config, "projection_size", None) or getattr(vision_config, "hidden_size")
        )

    def embed(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                inputs = {key: value.to(self.device) for key, value in self.processor(images=images[offset : offset + batch_size], return_tensors="pt").items()}
                output = self.model.get_image_features(**inputs)
                features = getattr(output, "pooler_output", output)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                chunks.append(features.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32, copy=False)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CatalogIndex:
    def __init__(self, embeddings: np.ndarray, entries: list[dict[str, Any]], metadata: dict[str, Any]):
        if embeddings.ndim != 2 or len(embeddings) != len(entries) or not len(entries):
            raise ValueError("index embeddings and entries are inconsistent or empty")
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported index schema")
        if metadata.get("embedding_dimensions") != embeddings.shape[1]:
            raise ValueError("index metadata dimensions do not match embeddings")
        required_metadata = {"catalog_version", "model_source", "model_revision", "gallery_policy"}
        if any(not metadata.get(key) for key in required_metadata):
            raise ValueError("index is missing model/catalog version metadata")
        if metadata["gallery_policy"] not in {"confirmed", "provisional"}:
            raise ValueError("unsupported gallery policy")
        if not np.isfinite(embeddings).all():
            raise ValueError("index contains non-finite embeddings")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=5e-3):
            raise ValueError("index embeddings are not L2-normalized")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.entries = entries
        self.metadata = metadata

    @classmethod
    def load(cls, path: Path) -> "CatalogIndex":
        with np.load(path, allow_pickle=False) as data:
            entries = json.loads(str(data["entries_json"].item()))
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(data["embeddings"], entries, metadata)

    def search(self, query_embeddings: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query embedding dimensions do not match index")
        if not np.isfinite(query_embeddings).all():
            raise ValueError("query contains non-finite embeddings")
        if not np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=5e-3):
            raise ValueError("query embeddings are not L2-normalized")
        scores = (query_embeddings @ self.embeddings.T).max(axis=0)
        best_by_slug: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, entry in zip(scores.tolist(), self.entries):
            slug = entry["slug"]
            if slug not in best_by_slug or score > best_by_slug[slug][0]:
                best_by_slug[slug] = (float(score), entry)
        ranked = sorted(best_by_slug.values(), key=lambda item: item[0], reverse=True)[:limit]
        return [
            {
                "slug": entry["slug"],
                "score": score,
                "card": entry["card"],
                "provenance": entry["provenance"],
            }
            for score, entry in ranked
        ]


def _save_index(path: Path, embeddings: np.ndarray, entries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(
            temporary,
            embeddings=embeddings,
            entries_json=np.array(json.dumps(entries, ensure_ascii=False)),
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_index(
    manifest_path: Path,
    output_path: Path,
    model_source: str,
    revision: str | None,
    device: str,
    batch_size: int,
    local_only: bool,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    repo_root = manifest_path.parent.parent
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(manifest_path)
    eligible = [record for record in manifest if is_index_eligible(record, repo_root, allow_provisional)]
    if not eligible:
        raise ValueError("no confirmed, ready, readable catalog images in manifest")
    encoder = ImageEncoder(model_source, revision, device, local_only)
    pending_images: list[Image.Image] = []
    embedding_chunks: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in eligible:
        path = manifest_image_path(record, repo_root)
        image = decode_image(path.read_bytes(), MAX_CATALOG_PIXELS)
        image.thumbnail((2048, 2048))
        for view_name, view in retrieval_views(image):
            pending_images.append(view)
            entries.append(
                {
                    "slug": manifest_slug(record),
                    "view": view_name,
                    "card": public_card(record),
                    "provenance": {
                        "identity_status": "confirmed" if _nested(record, "join", "confirmed") else "provisional",
                        "join_method": _nested(record, "join", "method"),
                    },
                }
            )
        if len(pending_images) >= batch_size:
            embedding_chunks.append(encoder.embed(pending_images, batch_size))
            for pending in pending_images:
                pending.close()
            pending_images.clear()
    if pending_images:
        embedding_chunks.append(encoder.embed(pending_images, batch_size))
        for pending in pending_images:
            pending.close()
    embeddings = np.concatenate(embedding_chunks)
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_digest:
        raise RuntimeError("manifest changed while the index was building")
    metadata = {
        "schema_version": 1,
        "catalog_version": manifest_digest[:16],
        "model_source": model_source,
        "model_revision": revision,
        "device_built_on": encoder.device,
        "manifest_records": len(manifest),
        "indexed_slugs": len({entry["slug"] for entry in entries}),
        "embedding_rows": len(entries),
        "embedding_dimensions": int(embeddings.shape[1]),
        "build_seconds": round(time.perf_counter() - started, 3),
        "coverage_complete": len(eligible) == len(manifest),
        "coverage_ratio": len(eligible) / len(manifest),
        "indexed_identity_status": dict(
            Counter("confirmed" if _nested(record, "join", "confirmed") else "provisional" for record in eligible)
        ),
        "gallery_policy": "provisional" if allow_provisional else "confirmed",
    }
    _save_index(output_path, embeddings, entries, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact cosine catalog index from confirmed assets")
    parser.add_argument("--manifest", type=Path, default=Path("data/catalog_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catalog_index.npz"))
    parser.add_argument("--model", default=os.getenv("WINE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--revision", default=os.getenv("WINE_MODEL_REVISION", DEFAULT_REVISION))
    parser.add_argument("--device", default=os.getenv("WINE_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-only", action="store_true", default=os.getenv("WINE_LOCAL_ONLY") == "1")
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="also index decoded candidate joins; ambiguity/missing remain excluded",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_index(
                args.manifest,
                args.output,
                args.model,
                args.revision,
                args.device,
                args.batch_size,
                args.local_only,
                args.allow_provisional,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
