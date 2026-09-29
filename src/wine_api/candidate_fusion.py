from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

CONFIG = {
    "schema_version": 1, "pool_size": 6, "fuzzy_token_min": 0.78,
    "rare_producer_df_max": 6, "rare_producer_similarity_min": 0.88,
    "multi_token_similarity_min": 0.80, "name_token_similarity_min": 0.85,
    "multi_token_field_score_min": 0.88,
    "replacement_score_margin": 0.05, "year_mismatch_penalty": 0.10,
    "minimum_qwen_score": 0.52, "name_match_override_margin": 0.05,
    "weights": {"producer": 0.35, "name": 0.50, "grapes": 0.15},
}
GENERIC = {
    "wine", "winery", "vineyard", "vineyards", "estate", "selection", "reserve",
    "вино", "винодельня", "виноград", "усадьба", "поместье", "резерв", "сухое",
    "полусухое", "полусладкое", "сладкое", "белое", "красное", "розовое", "брют",
    "игристое", "россия", "кубань", "crimea", "крым",
}
NAME_CATEGORY_TOKENS = {"российское", "шампанское", "выдержанное"}
PRODUCER_CATEGORY_TOKENS = {"семейная", "семейный", "семейное", "family"}
YEAR_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)(?!\d)")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def tokens(value: Any) -> list[str]:
    return [token for token in normalize(value).split() if len(token) >= 4 and token not in GENERIC and not token.isdigit()]


def years(value: Any) -> set[str]:
    text = re.sub(r"(?:цена|price)\s+\d+|\d+\s*(?:руб|rub|rur)", " ", normalize(value))
    return set(YEAR_RE.findall(text))


def token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
    shorter, longer = sorted((left, right), key=len)
    return max(similarity, 0.92) if len(shorter) >= 6 and longer.startswith(shorter) and len(shorter) / len(longer) >= 0.75 else similarity


def field_match(query_tokens: list[str], value: Any, min_similarity: float | None = None) -> dict[str, Any]:
    candidate = tokens(value)
    matches = []
    for token in candidate:
        similarity, query = max((token_similarity(token, item), item) for item in query_tokens) if query_tokens else (0.0, "")
        matches.append((token, query, similarity))
    strong = [item for item in matches if item[2] >= (min_similarity or CONFIG["multi_token_similarity_min"])]
    best = max((item[2] for item in matches), default=0.0)
    coverage = min(1.0, len(strong) / min(2, len(candidate))) if candidate else 0.0
    return {"score": 0.7 * best + 0.3 * coverage, "best": best, "strong": strong}


def load_catalog(path: Path, allowed_slugs: set[str] | None = None) -> tuple[dict[str, dict[str, Any]], Counter[str], dict[str, list[str]]]:
    catalog: dict[str, dict[str, Any]] = {}
    producer_names_by_token: dict[str, set[str]] = defaultdict(set)
    assets: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slug = str(row.get("wine_id") or row.get("slug") or "")
        if not slug or (allowed_slugs is not None and slug not in allowed_slugs):
            continue
        card = {
            "slug": slug, "name": row.get("canonical_name") or row.get("name"),
            "producer": row.get("producer"), "grapes": row.get("grapes"),
            "region": row.get("region"), "type": row.get("type"), "color": row.get("color"),
        }
        catalog[slug] = card
        producer_name = normalize(card["producer"])
        for token in set(tokens(card["producer"])):
            producer_names_by_token[token].add(producer_name)
        asset = str((row.get("image") or {}).get("sha256") or "")
        if asset:
            assets[asset].append(slug)
            card["asset_sha256"] = asset
    return catalog, Counter({token: len(names) for token, names in producer_names_by_token.items()}), assets


def score_card(ocr_text: str, card: dict[str, Any], producer_df: Counter[str]) -> dict[str, Any]:
    query_tokens = tokens(ocr_text)
    fields = {
        field: field_match(query_tokens, card.get(field), CONFIG["name_token_similarity_min"] if field == "name" else None)
        for field in ("producer", "name", "grapes")
    }
    rare = [
        (token, query, similarity) for token, query, similarity in fields["producer"]["strong"]
        if producer_df[token] <= CONFIG["rare_producer_df_max"] and similarity >= CONFIG["rare_producer_similarity_min"]
    ]
    admitted_by = ["rare_fuzzy_producer"] if rare else []
    for field in ("name", "grapes"):
        if len(fields[field]["strong"]) >= 2 and fields[field]["score"] >= CONFIG["multi_token_field_score_min"]:
            admitted_by.append(f"two_token_{field}")
    query_years, card_years = years(ocr_text), years(" ".join(str(value or "") for value in card.values()))
    year_match = bool(query_years and card_years and not query_years.isdisjoint(card_years))
    year_mismatch = bool(query_years and card_years and query_years.isdisjoint(card_years))
    weighted = sum(CONFIG["weights"][field] * fields[field]["score"] for field in fields)
    producer_tokens = set(tokens(card.get("producer")))
    exact_name_tokens = {
        token for token, _, similarity in fields["name"]["strong"]
        if similarity == 1.0 and token not in NAME_CATEGORY_TOKENS and token not in producer_tokens
    }
    evidence = [
        f"{field}:{catalog_token}~{ocr_token}:{similarity:.3f}"
        for field, result in fields.items() for catalog_token, ocr_token, similarity in result["strong"]
    ] + admitted_by
    return {
        "ocr_entity_score": round(max(0.0, weighted - CONFIG["year_mismatch_penalty"] * year_mismatch), 6),
        "ocr_name_exact_matches": len(exact_name_tokens),
        "ocr_year_match": year_match, "ocr_year_mismatch": year_mismatch,
        "admitted_by": admitted_by, "evidence": evidence,
    }


def similarity_reason(
    ocr_text: str, card: dict[str, Any], qwen_score: float,
    catalog: dict[str, dict[str, Any]], producer_df: Counter[str],
) -> str | None:
    if qwen_score < CONFIG["minimum_qwen_score"]:
        return "low_reranker_score"
    grape_tokens = {token for item in catalog.values() for token in tokens(item.get("grapes"))}
    producer_tokens = {
        token for token in tokens(ocr_text)
        if 0 < producer_df[token] <= CONFIG["rare_producer_df_max"]
        and token not in grape_tokens and token not in PRODUCER_CATEGORY_TOKENS
    }
    card_tokens = set(tokens(card.get("producer"))) | set(tokens(card.get("name")))
    if producer_tokens and producer_tokens.isdisjoint(card_tokens):
        return "ocr_producer_conflict"
    return None


def prefer_exact_name(candidates: list[dict[str, Any]]) -> bool:
    best_qwen = max(item["reranker_score"] for item in candidates)
    candidates.sort(key=lambda item: (
        not (
            item.get("ocr_name_exact_matches", 0) >= 2
            and best_qwen - item["reranker_score"] <= CONFIG["name_match_override_margin"]
        ),
        -item["reranker_score"], item["slug"],
    ))
    return candidates[0]["reranker_score"] < best_qwen


def fuse(visual: list[dict[str, Any]], ocr_text: str, catalog: dict[str, dict[str, Any]], producer_df: Counter[str], assets: dict[str, list[str]]) -> list[dict[str, Any]]:
    scored = {slug: score_card(ocr_text, card, producer_df) for slug, card in catalog.items()}
    pool = [
        {"slug": item["slug"], "visual_rank": rank, "visual_score": float(item["score"]), "source": "visual", **scored[item["slug"]]}
        for rank, item in enumerate(visual[: CONFIG["pool_size"]], 1) if item["slug"] in catalog
    ]
    visual_slugs = {item["slug"] for item in pool}
    outsiders = [
        ("two_token_name" in result["admitted_by"], result["ocr_entity_score"], slug)
        for slug, result in scored.items() if slug not in visual_slugs and result["admitted_by"]
    ]
    slug = ""
    if outsiders and len(pool) == CONFIG["pool_size"]:
        _, best_score, slug = max(outsiders)
        if best_score < pool[-1]["ocr_entity_score"] + CONFIG["replacement_score_margin"]:
            slug = ""
    if slug:
        pool[-1] = {
            "slug": slug,
            "visual_rank": next((rank for rank, item in enumerate(visual, 1) if item["slug"] == slug), None),
            "visual_score": next((float(item["score"]) for item in visual if item["slug"] == slug), None),
            "source": "ocr_catalog_admission", **scored[slug],
        }
    for item in pool:
        asset = str(catalog[item["slug"]].get("asset_sha256") or "")
        peers = assets.get(asset, []) if asset else []
        if len(peers) > 1:
            item["evidence"].append(f"shared_asset_metadata_variants:{len(peers)}")
    return pool
