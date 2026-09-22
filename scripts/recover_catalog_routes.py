#!/usr/bin/env python3
"""Propose moved public routes for the final catalog 404s; never apply them."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_catalog_assets import BASE, USER_AGENT, Fetcher, load_robots, parse_card, sha256  # noqa: E402


CSV_PATH = Path("catalog.csv")
CANDIDATES = {
    "kokur-suhoe-2025": ["kokur-2025"],
    "method-classic-kokur": ["pino-nuar-2025"],
    "ona-skazala-da": ["fanagoriya-ona-skazala-da-beloe-bryut",
                        "fanagoriya-ona-skazala-da-beloe-polusladkoe"],
    "pinot-noir-2024-one-barrel-by-dmitry-maslov-pino-nuar-2024-uan-barrel-dmitrij-maslov":
        ["chardonnay-2024", "pinot-noir-2024"],
    "polusladkoe-krasnoe-zb-vajn-frizzante":
        ["zolotaya-balka-igristoe-krasnoe-polusladkoe"],
    "polusladkoe-krasnoe-zolotaya-balka":
        ["zolotaya-balka-igristoe-krasnoe-polusladkoe"],
    "vibes-silvaner-barrel-fermented-2022":
        ["vibes-vermentino-viognier-barrel-fermented-2022"],
}


def norm(value: str) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", value.lower().replace("ё", "е")))


def photo_matches(photo: str, asset: str) -> bool:
    photo_stem = norm(Path(photo).stem).replace(" ", "")
    asset_stem = re.sub(r"_[0-9a-f]{10}$", "", norm(Path(asset).stem).replace(" ", ""))
    return len(photo_stem) >= 6 and (photo_stem in asset_stem or asset_stem in photo_stem)


def main() -> None:
    patches_path = Path("artifacts/catalog_v2/source_patches.jsonl")
    output = Path("artifacts/catalog_route_recovery/proposals.jsonl")
    raw_dir = output.parent / "raw_cards"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [row["slug"] for row in map(json.loads, patches_path.read_text().splitlines())
               if row["status"] == "page_not_found"]
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as source:
        csv_rows = {row["Slug"]: row for row in csv.DictReader(source)}

    fetcher = Fetcher(timeout=15.0)
    robots = load_robots(fetcher, BASE)
    sitemap_url = f"{BASE}/wines-sitemap.xml"
    sitemap_bytes, final_url = fetcher.get(sitemap_url)
    if final_url != sitemap_url:
        raise ValueError("unexpected sitemap redirect")
    root = ET.fromstring(sitemap_bytes)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9",
          "i": "http://www.google.com/schemas/sitemap-image/1.1"}
    sitemap = {}
    for item in root.findall("s:url", ns):
        url = item.findtext("s:loc", "", ns)
        route = url.rsplit("/", 1)[-1]
        sitemap[route] = {
            "route": route,
            "source_url": url,
            "sitemap_title": item.findtext("i:image/i:title", "", ns),
            "sitemap_image_url": item.findtext("i:image/i:loc", "", ns),
        }

    fetched = {}
    for route in sorted({route for routes in CANDIDATES.values() for route in routes}):
        entry = sitemap.get(route)
        if not entry or not robots.can_fetch(USER_AGENT, entry["source_url"]):
            continue
        data, card_final_url = fetcher.get(entry["source_url"])
        if card_final_url != entry["source_url"]:
            continue
        card = parse_card(data.decode("utf-8"), route)
        digest = sha256(data)
        raw_path = raw_dir / f"{digest}.html"
        if not raw_path.exists():
            raw_path.write_bytes(data)
        fetched[route] = {**entry, **card, "final_url": card_final_url,
                          "raw_html_sha256": digest, "raw_html_path": raw_path.resolve().as_posix(),
                          "fetched_at": datetime.now(timezone.utc).isoformat(),
                          "html_normalized": norm(data.decode("utf-8"))}

    proposals = []
    for slug in targets:
        source = csv_rows[slug]
        checks = []
        for route in CANDIDATES.get(slug, []):
            card = fetched.get(route)
            if not card:
                continue
            html = card["html_normalized"]
            winery = norm(source["Винодельня"])
            grapes = [norm(value) for value in source["Сорт винограда"].split(",") if norm(value)]
            years = re.findall(r"20\d{2}", source["Название вина"])
            name_score = SequenceMatcher(None, norm(source["Название вина"]), norm(card["name"])).ratio()
            evidence = {
                "name_similarity": round(name_score, 4),
                "winery_match": bool(winery and winery in html),
                "grape_matches": {grape: grape in html for grape in grapes},
                "year_matches": {year: year in html for year in years},
                "source_photo_matches_asset": photo_matches(source["Название фото"], card["asset"]),
            }
            conflicts = []
            if not evidence["winery_match"]:
                conflicts.append("winery")
            if any(not value for value in evidence["grape_matches"].values()):
                conflicts.append("grape")
            if any(not value for value in evidence["year_matches"].values()):
                conflicts.append("year")
            if name_score < 0.55:
                conflicts.append("name")
            confidence = "strong" if not conflicts else "ambiguous"
            checks.append({k: v for k, v in card.items() if k != "html_normalized"}
                          | {"evidence": evidence, "confidence": confidence,
                             "conflicts": conflicts})
        strong = [row for row in checks if row["confidence"] == "strong"]
        proposals.append({
            "target_slug": slug,
            "target_name": source["Название вина"],
            "target_winery": source["Винодельня"],
            "target_grapes": source["Сорт винограда"],
            "target_photo": source["Название фото"],
            "status": "strong_moved_route_candidate" if len(strong) == 1 else
                      ("ambiguous_candidates" if checks else "no_strong_sitemap_candidate"),
            "recommended": strong[0] if len(strong) == 1 else None,
            "checked_candidates": checks,
            "applied": False,
            "provenance": "official_public_wines_sitemap_plus_card_ssr_cross_checked_against_csv",
        })

    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in proposals),
                      encoding="utf-8")
    summary = {
        "targets": len(proposals),
        "strong": sum(row["status"] == "strong_moved_route_candidate" for row in proposals),
        "ambiguous": sum(row["status"] == "ambiguous_candidates" for row in proposals),
        "no_candidate": sum(row["status"] == "no_strong_sitemap_candidate" for row in proposals),
        "card_requests": len(fetched),
        "total_requests": len(fetched) + 2,
        "sitemap_sha256": hashlib.sha256(sitemap_bytes).hexdigest(),
    }
    (output.parent / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
