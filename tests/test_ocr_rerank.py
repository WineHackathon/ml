import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts/ocr_rerank.py"
SPEC = importlib.util.spec_from_file_location("ocr_rerank", SCRIPT)
ocr_rerank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ocr_rerank)
normalize, rerank, years = ocr_rerank.normalize, ocr_rerank.rerank, ocr_rerank.years


def test_normalization_years_and_explicit_ocr_device(monkeypatch, tmp_path):
    assert normalize("ПИНО-НУАР, Ёлка") == "пино нуар елка"
    assert years("цена 1990 руб; урожай 2021") == {"2021"}
    assert years("цена 48.98; крепость 13.5") == set()
    calls = []
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=lambda **kwargs: calls.append(kwargs)))
    monkeypatch.delenv("WINE_OCR_DEVICE", raising=False)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    digest = hashlib.sha256()
    digest.update(b"image")
    digest.update(json.dumps(ocr_rerank.OCR_CONFIG, sort_keys=True).encode())
    legacy_key = digest.hexdigest()
    digest.update(json.dumps({"backend": "cpu", "enable_mkldnn": False}, sort_keys=True).encode())
    cpu_key = ocr_rerank._cache_key(image)
    assert cpu_key == digest.hexdigest()
    assert cpu_key != legacy_key
    ocr_rerank.load_engine(4)
    monkeypatch.setenv("WINE_OCR_DEVICE", "gpu:0")
    gpu_key = ocr_rerank._cache_key(image)
    ocr_rerank.load_engine(4)
    monkeypatch.setenv("WINE_OCR_DEVICE", "gpu:1")
    assert gpu_key != cpu_key
    assert ocr_rerank._cache_key(image) == gpu_key
    assert [(call["device"], call["enable_mkldnn"]) for call in calls] == [("cpu", False), ("gpu:0", False)]


def test_missing_year_is_neutral_but_conflicting_year_is_penalized():
    visual = [
        {"slug": "wrong", "score": 0.80, "card": {"slug": "wrong", "name": "Мерло 2020"}},
        {"slug": "missing", "score": 0.79, "card": {"slug": "missing", "name": "Мерло"}},
        {"slug": "right", "score": 0.78, "card": {"slug": "right", "name": "Мерло 2021"}},
    ]
    ranked, _ = rerank(visual, "Мерло 2021", {item["slug"]: item["card"] for item in visual})
    by_slug = {item["slug"]: item for item in ranked}
    assert by_slug["wrong"]["year_mismatch"] is True
    assert by_slug["missing"]["year_mismatch"] is False
    assert ranked[0]["slug"] == "right"


def test_strong_gallery_text_match_can_enter_without_labels():
    visual = [{"slug": "visual", "score": 0.80, "card": {"slug": "visual", "name": "Other Wine"}}]
    gallery = {
        "visual": visual[0]["card"],
        "ocr-hit": {"slug": "ocr-hit", "name": "Denisov Красная Стрелка", "producer": "Denisov"},
    }
    ranked, _ = rerank(visual, "DENISOV КРАСНАЯ СТРЕЛКА", gallery)
    assert ranked[0]["slug"] == "ocr-hit"
    assert ranked[0]["source"] == "gallery_text"
