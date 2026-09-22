import importlib.util
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/fetch_catalog_assets.py"
SPEC = importlib.util.spec_from_file_location("fetch_catalog_assets", SCRIPT)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class CatalogAssetTests(unittest.TestCase):
    def test_parses_only_matching_official_wine_image(self):
        html = '''<meta property="og:title" content="Wine 2025">
        <meta property="og:image" content="https://api.vino-svoe.ru/v1/img/str-api/1200/630/resize/uploads/Wine_a1b2c3d4e5.webp">
        <script id="__NUXT_DATA__" type="application/json">[["wine-wine-2025"],"wine-2025"]</script>'''
        self.assertEqual(assets.parse_card(html, "wine-2025")["asset"], "Wine_a1b2c3d4e5.webp")
        with self.assertRaises(ValueError):
            assets.parse_card(html.replace("api.vino-svoe.ru", "example.com"), "wine-2025")

    def test_filters_unresolved_and_matches_exact_original_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in [
                {"wine_id": "ready", "status": {"catalog_image": "ready"}},
                {"wine_id": "missing", "status": {"catalog_image": "missing"}},
                {"wine_id": "ambiguous", "status": {"catalog_image": "ambiguous"}},
            ]), encoding="utf-8")
            self.assertEqual([row["wine_id"] for row in assets.unresolved(manifest)],
                             ["missing", "ambiguous"])
            inventory = root / "inventory.jsonl"
            inventory.write_text(
                json.dumps({"member": "uploads/Wine_a1b2c3d4e5.webp", "kind": "original_candidate"})
                + "\n" + json.dumps({"member": "uploads/thumbnail_Wine_a1b2c3d4e5.webp",
                                      "kind": "cms_derivative"}) + "\n", encoding="utf-8")
            self.assertEqual(assets.archive_assets(inventory),
                             {"Wine_a1b2c3d4e5.webp": ["uploads/Wine_a1b2c3d4e5.webp"]})

    def test_robots_wildcard_and_image_validation(self):
        robots = assets.RobotsRules("User-agent: *\nDisallow: /\nAllow: */img/*\n")
        self.assertTrue(robots.can_fetch(assets.USER_AGENT,
                                        "https://api.vino-svoe.ru/v1/img/resize/uploads/wine.webp"))
        self.assertFalse(robots.can_fetch(assets.USER_AGENT,
                                         "https://api.vino-svoe.ru/uploads/wine.webp"))
        buffer = BytesIO()
        Image.new("RGB", (9, 13), "red").save(buffer, "WEBP")
        self.assertEqual(assets.verify_image(buffer.getvalue()), ("webp", 9, 13))

    def test_candidate_asset_is_corroborated_only_by_same_member(self):
        self.assertEqual(
            assets.matched_asset_status("candidate", "same.webp", ["uploads/same.webp"]),
            "corroborated_manifest_asset",
        )
        self.assertEqual(
            assets.matched_asset_status("candidate", "current.webp", ["uploads/old.webp"]),
            "changed_official_archive_asset",
        )


if __name__ == "__main__":
    unittest.main()
