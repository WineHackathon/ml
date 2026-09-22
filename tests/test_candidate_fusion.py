from collections import Counter

from wine_api import candidate_fusion as fusion


def test_fuzzy_rare_producer_admits_catalog_candidate():
    catalog = {
        **{f"visual-{i}": {"slug": f"visual-{i}", "name": "Other", "producer": "Common", "grapes": "Merlot"} for i in range(6)},
        "target": {"slug": "target", "name": "Мерло", "producer": "Редковин", "grapes": "Мерло"},
    }
    visual = [{"slug": f"visual-{i}", "score": 1 - i / 100} for i in range(6)]
    pool = fusion.fuse(visual, "Семейная винодельня Редковина", catalog, Counter({"редковин": 1, "common": 20}), {})
    assert pool[-1]["slug"] == "target"
    assert pool[-1]["source"] == "ocr_catalog_admission"
    assert "rare_fuzzy_producer" in pool[-1]["admitted_by"]


def test_two_grape_tokens_distinguish_metadata_variants():
    query = "АЛИГОТЕ ЦИТРОН БЕЛОЕ СУХОЕ 2024"
    correct = fusion.score_card(query, {"name": "Жемчужная Алиготе Цитрон", "producer": "АРАТТИ", "grapes": "Алиготе Цитронный Магарача"}, Counter())
    wrong = fusion.score_card(query, {"name": "Жемчужная Цитрон Шардоне", "producer": "АРАТТИ", "grapes": "Цитронный Магарача Шардоне"}, Counter())
    assert "two_token_grapes" in correct["admitted_by"]
    assert correct["ocr_entity_score"] > wrong["ocr_entity_score"]


def test_equal_ocr_evidence_does_not_displace_visual_rank_six():
    catalog = {f"visual-{i}": {"slug": f"visual-{i}", "name": "Rare Name", "producer": "Rare", "grapes": ""} for i in range(6)}
    catalog["outsider"] = {"slug": "outsider", "name": "Rare Name", "producer": "Rare", "grapes": ""}
    visual = [{"slug": f"visual-{i}", "score": 1 - i / 100} for i in range(6)]
    pool = fusion.fuse(visual, "Rare Name", catalog, Counter({"rare": 1}), {})
    assert pool[-1]["slug"] == "visual-5"


def test_price_is_not_vintage_and_missing_year_is_neutral():
    assert fusion.years("цена 1990 руб, 48.98") == set()
    scored = fusion.score_card("Рислинг 2023", {"name": "Рислинг", "producer": "", "grapes": "Рислинг"}, Counter())
    assert scored["ocr_year_match"] is False
    assert scored["ocr_year_mismatch"] is False
