import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/ocr_rerank.py"
SPEC = importlib.util.spec_from_file_location("ocr_rerank", SCRIPT)
ocr_rerank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ocr_rerank)
normalize, rerank, years = ocr_rerank.normalize, ocr_rerank.rerank, ocr_rerank.years


def test_normalization_and_years_do_not_promote_prices():
    assert normalize("ПИНО-НУАР, Ёлка") == "пино нуар елка"
    assert years("цена 1990 руб; урожай 2021") == {"2021"}
    assert years("цена 48.98; крепость 13.5") == set()


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
