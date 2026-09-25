#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image


OCR_CONFIG = {
    "detector": "PP-OCRv5_mobile_det",
    "recognizer": "eslav_PP-OCRv5_mobile_rec",
    "crop": [0.05, 0.18, 0.95, 0.94],
    "max_side": 1600,
    "min_confidence": 0.45,
}
OCR_WEIGHT = 0.12
YEAR_MATCH_BONUS = 0.03
YEAR_MISMATCH_PENALTY = 0.15
TEXT_ONLY_VISUAL_GAP = 0.10
TEXT_CANDIDATE_MIN = 0.86
GENERIC = {
    "wine", "winery", "вино", "винодельня", "сухое", "белое", "красное",
    "розовое", "брют", "reserve", "резерв", "россия", "кубань",
}
YEAR_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)(?!\d)")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def years(value: Any) -> set[str]:
    text = normalize(value)
    text = re.sub(r"(?:цена|price)\s+\d+|\d+\s*(?:руб|rub|rur)", " ", text)
    return set(YEAR_RE.findall(text))


def catalog_text(card: dict[str, Any]) -> str:
    return normalize(" ".join(str(card.get(key) or "") for key in ("name", "producer", "grapes", "slug")))


def text_score(query_text: str, candidate_text: str) -> tuple[float, set[str]]:
    query = set(normalize(query_text).split())
    candidate = set(normalize(candidate_text).split())
    informative = {token for token in query if len(token) >= 4 and token not in GENERIC and not token.isdigit()}
    shared = informative & candidate
    if not informative:
        return 0.0, set()
    coverage = len(shared) / len(informative)
    precision = len(shared) / max(1, len({token for token in candidate if len(token) >= 4}))
    return 0.8 * coverage + 0.2 * min(1.0, precision * 3), shared


def rerank(
    visual: list[dict[str, Any]],
    ocr_text: str,
    gallery_cards: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not visual:
        return [], []
    query_years = years(ocr_text)
    scored: dict[str, dict[str, Any]] = {}
    for item in visual:
        card = item.get("card") or gallery_cards.get(item["slug"], {})
        match, shared = text_score(ocr_text, catalog_text(card))
        candidate_years = years(catalog_text(card))
        year_match = bool(query_years and candidate_years and not query_years.isdisjoint(candidate_years))
        mismatch = bool(query_years and candidate_years and query_years.isdisjoint(candidate_years))
        scored[item["slug"]] = {
            **item,
            "ocr_text_score": round(match, 6),
            "shared_tokens": sorted(shared),
            "year_mismatch": mismatch,
            "combined_score": float(item["score"]) + OCR_WEIGHT * match + YEAR_MATCH_BONUS * year_match - YEAR_MISMATCH_PENALTY * mismatch,
            "source": "visual",
        }

    text_ranked: list[dict[str, Any]] = []
    for slug, card in gallery_cards.items():
        match, shared = text_score(ocr_text, catalog_text(card))
        candidate_years = years(catalog_text(card))
        mismatch = bool(query_years and candidate_years and query_years.isdisjoint(candidate_years))
        text_ranked.append({
            "slug": slug,
            "score": round(match, 6),
            "shared_tokens": sorted(shared),
            "year_mismatch": mismatch,
        })
    text_ranked.sort(key=lambda item: (-item["score"], item["slug"]))

    for candidate in text_ranked[:5]:
        shared = candidate["shared_tokens"]
        distinctive = len(shared) >= 2 or any(len(token) >= 8 for token in shared)
        if (
            candidate["slug"] not in scored
            and candidate["score"] >= TEXT_CANDIDATE_MIN
            and distinctive
            and not candidate["year_mismatch"]
        ):
            card = gallery_cards[candidate["slug"]]
            proxy_visual = float(visual[0]["score"]) - TEXT_ONLY_VISUAL_GAP
            scored[candidate["slug"]] = {
                "slug": candidate["slug"],
                "score": proxy_visual,
                "card": card,
                "provenance": {"identity_status": "gallery_text_candidate"},
                "ocr_text_score": candidate["score"],
                "shared_tokens": shared,
                "year_mismatch": False,
                "combined_score": proxy_visual + OCR_WEIGHT * candidate["score"],
                "source": "gallery_text",
            }

    ranked = sorted(scored.values(), key=lambda item: (-item["combined_score"], item["slug"]))[:5]
    return ranked, text_ranked[:5]


def _cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(json.dumps(OCR_CONFIG, sort_keys=True).encode())
    backend = os.getenv("WINE_OCR_DEVICE", "cpu").partition(":")[0]
    digest.update(json.dumps({"backend": backend, "enable_mkldnn": False}, sort_keys=True).encode())
    return digest.hexdigest()


def load_engine(cpu_threads: int):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=OCR_CONFIG["detector"],
        text_recognition_model_name=OCR_CONFIG["recognizer"],
        device=os.getenv("WINE_OCR_DEVICE", "cpu"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        cpu_threads=cpu_threads,
        enable_mkldnn=False,
    )


def predict_image(engine, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        left, top, right, bottom = OCR_CONFIG["crop"]
        box = (int(width * left), int(height * top), int(width * right), int(height * bottom))
        crop = image.crop(box)
        crop.thumbnail((OCR_CONFIG["max_side"], OCR_CONFIG["max_side"]))
    with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
        crop.save(handle.name, quality=92)
        result = list(engine.predict(handle.name))[0].json["res"]
    pairs = [
        [str(text), round(float(score), 6)]
        for text, score in zip(result["rec_texts"], result["rec_scores"])
        if float(score) >= OCR_CONFIG["min_confidence"]
    ]
    return {
        "text": " ".join(text for text, _ in pairs),
        "lines": pairs,
        "crop_pixels": list(box),
        "source_size": [width, height],
        "seconds": round(time.perf_counter() - started, 3),
        "config": OCR_CONFIG,
    }


def extract(paths: list[Path], cache_path: Path, cpu_threads: int, telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        cache = {}
    telemetry.update(cache.get("_meta", {}))
    keys = {str(path): _cache_key(path) for path in paths}
    pending = [path for path in paths if keys[str(path)] not in cache]
    if pending:
        loading = time.perf_counter()
        engine = load_engine(cpu_threads)
        telemetry["model_load_seconds"] = round(time.perf_counter() - loading, 3)
        for path in pending:
            cache[keys[str(path)]] = predict_image(engine, path)
        cache["_meta"] = telemetry
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, cache_path)
    return {path: cache[keys[path]] for path in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached label-crop OCR for the offline proxy experiment")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ocr_cache.json"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    telemetry: dict[str, Any] = {}
    results = extract(args.images, args.cache, args.cpu_threads, telemetry)
    telemetry["batch_wall_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps({"results": results, "timings": telemetry}, ensure_ascii=False))


if __name__ == "__main__":
    main()
