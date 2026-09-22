import hashlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    "apply_catalog_asset_patches", Path(__file__).parents[1] / "scripts/apply_catalog_asset_patches.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_applies_only_unambiguous_official_asset(tmp_path):
    image_path = tmp_path / "source.webp"
    Image.new("RGB", (7, 11), "red").save(image_path, "WEBP")
    data = image_path.read_bytes()
    slug = "wine-one"
    raw_path = tmp_path / "card.html"
    raw_path.write_text("official card", encoding="utf-8")
    manifest = [{
        "wine_id": slug, "canonical_name": "Wine One", "image": None,
        "source": {"csv": "source.csv", "member": None},
        "join": {"confirmed": False},
        "status": {"catalog_image": "missing", "training_eligible": False},
    }]
    patch = {
        "slug": slug, "asset": "wine.webp", "status": "resolved_archive_exact_asset",
        "source_url": f"https://vino-svoe.ru/wines/{slug}",
        "final_url": f"https://vino-svoe.ru/wines/{slug}",
        "image_url": "https://api.vino-svoe.ru/uploads/wine.webp",
        "archive_matches": ["uploads/wine.webp"],
        "provenance": "official_public_card_ssr_and_og_image",
        "raw_html_path": str(raw_path),
        "raw_html_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }

    repaired, receipts = MODULE.apply_patches(manifest, [patch], tmp_path / "assets", lambda _: data)

    assert repaired[0]["source"] == manifest[0]["source"]
    assert repaired[0]["join"]["confirmed"] is True
    assert repaired[0]["status"] == {"catalog_image": "ready", "training_eligible": True, "reasons": []}
    assert repaired[0]["image"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert receipts[0]["pixels"] == [7, 11]


def test_rejects_ambiguous_patch(tmp_path):
    manifest = [{"wine_id": "wine", "status": {"catalog_image": "missing"}}]
    patch = {"slug": "wine", "status": "resolved_archive_exact_asset", "archive_matches": ["a", "b"]}
    with pytest.raises(ValueError, match="unverified or ambiguous"):
        MODULE.apply_patches(manifest, [patch], tmp_path, lambda _: b"")


def test_applies_verified_public_asset_and_reports_aliases(tmp_path):
    source = tmp_path / "source.webp"
    Image.new("RGB", (5, 6), "blue").save(source, "WEBP")
    raw = tmp_path / "card.html"
    raw.write_text("official", encoding="utf-8")
    data = source.read_bytes()
    manifests = [{
        "wine_id": slug, "canonical_name": slug, "image": None,
        "join": {"confirmed": False},
        "status": {"catalog_image": "missing", "training_eligible": False},
    } for slug in ("one", "two")]
    patches = [{
        "slug": slug, "asset": "shared.webp", "status": "resolved_official_public_asset",
        "source_url": f"https://vino-svoe.ru/wines/{slug}",
        "final_url": f"https://vino-svoe.ru/wines/{slug}",
        "image_url": "https://api.vino-svoe.ru/v1/img/uploads/shared.webp",
        "archive_matches": [], "asset_path": str(source),
        "asset_sha256": hashlib.sha256(data).hexdigest(), "asset_pixels": [5, 6],
        "provenance": "official_public_card_ssr_and_og_image",
        "raw_html_path": str(raw), "raw_html_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    } for slug in ("one", "two")]

    repaired, receipts = MODULE.apply_patches(manifests, patches, tmp_path / "assets", lambda _: b"")
    manifest_path, patches_path = tmp_path / "manifest.jsonl", tmp_path / "patches.jsonl"
    manifest_path.write_text("\n".join(map(str, repaired)), encoding="utf-8")
    patches_path.write_text("\n".join(map(str, patches)), encoding="utf-8")
    report = MODULE.build_source_report(manifests, repaired, patches, receipts,
                                        manifest_path, patches_path)

    assert report["counts"]["applied_repairs"] == 2
    assert report["identical_asset_groups"][0]["slugs"] == ["one", "two"]
