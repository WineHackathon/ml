from collections import Counter
import json
from pathlib import Path

from wine_api import candidate_fusion as fusion

MUSCATEL_OCR = (
    "ПРОИЗВОДСТВЕНИ MACCAHAPA APA MACC 25.0.1 .26 год 1894 ВИНОРОССИИ "
    "МУСКАТЕЛЬ MYCKAT БИНО b KPACHOT MACCAHAPA PA БЕЛЫЙ ГОД УРОЖАЯ 2023 "
    "0,751 Собственные 0,75л Собетвенные винораднии Ha"
)


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


def test_real_muscatel_ocr_admits_exact_catalog_product():
    catalog, producer_df, assets = fusion.load_catalog(
        Path(__file__).resolve().parents[1] / "data/catalog_manifest.jsonl"
    )
    visual = [
        {"slug": slug, "score": 1 - rank / 100}
        for rank, slug in enumerate([
            "massandra-muskat",
            "massandra-portveyn-belyy-gurzuf-kokur-belyy-beloe-sladkoe-135",
            "massandra-muskat-belyy-yuzhnoberezhnyy-beloe-sladkoe-16",
            "massandra-heres",
            "massandra-muskatel-chernyy-krasnye-sorta-vinograda-krasnoe-sladkoe-16",
            "massandra-kokur",
        ])
    ]
    pool = fusion.fuse(visual, MUSCATEL_OCR, catalog, producer_df, assets)
    assert pool[-1]["slug"] == "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"
    assert pool[-1]["source"] == "ocr_catalog_admission"
    assert pool[-1]["ocr_name_exact_matches"] == 2


def test_generic_new_world_words_do_not_override_product_identity():
    catalog, producer_df, _ = fusion.load_catalog(
        Path(__file__).resolve().parents[1] / "data/catalog_manifest.jsonl"
    )
    ocr_text = "18 78 ДОМ ШАМПАНСКИХ ВИН tieste Новый Свъть ВЫДЕРЖАННОЕ 2023"
    wrong = fusion.score_card(
        ocr_text,
        catalog["novyy-svet-dom-shampanskih-vin-rossiyskoe-shampanskoe-vyderzhannoe-polusladkoe-rozovoe-novyy-svet-shardone-125"],
        producer_df,
    )
    assert wrong["ocr_name_exact_matches"] < 2


def test_exact_name_override_is_limited_to_close_qwen_scores():
    muscatel = {"slug": "muscatel", "reranker_score": 0.689137, "ocr_name_exact_matches": 2}
    muscat = {"slug": "muscat", "reranker_score": 0.735897, "ocr_name_exact_matches": 0}
    candidates = [muscat, muscatel]
    assert fusion.prefer_exact_name(candidates)
    assert candidates[0]["slug"] == "muscatel"

    candidates = [muscat, {**muscatel, "reranker_score": 0.680}]
    assert not fusion.prefer_exact_name(candidates)
    assert candidates[0]["slug"] == "muscat"


def test_ocr_cannot_admit_a_catalog_row_without_an_indexed_gallery(tmp_path):
    allowed = {f"visual-{rank}" for rank in range(6)}
    rows = [
        {"wine_id": slug, "canonical_name": "Other", "producer": "Common", "grapes": "Merlot"}
        for slug in sorted(allowed)
    ] + [{
        "wine_id": "method-classic-kokur", "canonical_name": "Method Classic Кокур",
        "producer": "Братья Мельниковы", "grapes": "Кокур",
    }]
    manifest = tmp_path / "catalog.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    catalog, producer_df, assets = fusion.load_catalog(manifest, allowed)
    visual = [{"slug": slug, "score": 1.0} for slug in sorted(allowed)]
    pool = fusion.fuse(visual, "Винодельня Братьев Мельниковых Method Classic Кокур", catalog, producer_df, assets)
    assert len(pool) == 6
    assert {item["slug"] for item in pool} == allowed


def test_real_eval_weak_matches_are_marked_as_similar():
    catalog, producer_df, _ = fusion.load_catalog(
        Path(__file__).resolve().parents[1] / "data/catalog_manifest.jsonl"
    )
    assert fusion.similarity_reason(
        "ТАБИЯ ВИНОДЕЛЬНЯ ПиноHуар полусухое 2025 5",
        catalog["usadba-mezyb-shishka-pino-nuar-rozovoe-suhoe-115"],
        0.626, catalog, producer_df,
    ) == "ocr_producer_conflict"
    assert fusion.similarity_reason(
        "ARISTOV БРЮТ 2023",
        catalog["aristov-anima-millesimato-beloe-bryut"],
        0.499, catalog, producer_df,
    ) == "low_reranker_score"
    assert fusion.similarity_reason(
        MUSCATEL_OCR,
        catalog["massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"],
        0.736, catalog, producer_df,
    ) is None
    assert fusion.similarity_reason(
        "GOLUBITSKOE —ESTATE CHARDONNAY TAMAGNE 2024",
        catalog["golubitskoe-estate-chardonnay"],
        0.7, catalog, producer_df,
    ) is None
    assert fusion.similarity_reason(
        "Семейная винодельня Литавщуков cухое красное",
        catalog["merlo-litavshhuk"],
        0.715, catalog, producer_df,
    ) is None
