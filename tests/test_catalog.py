import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_catalog.py"
SPEC = importlib.util.spec_from_file_location("prepare_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def test_normalization_preserves_numbers_and_removes_only_strapi_hash(self):
        self.assertEqual(catalog.normalize_name("Вионье 2023_a1b2c3d4e5.webp"), "vione_2023")
        self.assertEqual(catalog.normalize_name("01-Riesling (1).webp"), "01_riesling_1")

    def test_archive_path_guard(self):
        self.assertTrue(catalog.is_safe_member("uploads/wine.webp"))
        self.assertFalse(catalog.is_safe_member("../wine.webp"))
        self.assertFalse(catalog.is_safe_member("/tmp/wine.webp"))
        self.assertFalse(catalog.is_safe_member("C:\\wine.webp"))
        members = ["one/same.webp", "two/same.webp"]
        self.assertNotEqual(Path("extracted") / members[0], Path("extracted") / members[1])

    def test_filename_evidence_is_candidate_until_manually_verified(self):
        rows = {"photo.webp": ["same-as-photo"]}
        dual = catalog.decide_join(
            {"Название фото": "photo.webp", "Slug": "photo"}, ["uploads/photo_a1b2c3d4e5.webp"], rows)
        self.assertFalse(dual["confirmed"])
        self.assertEqual(dual["status"], "candidate")
        candidate = catalog.decide_join(
            {"Название фото": "Фото.webp", "Slug": "different"},
            ["uploads/Foto_a1b2c3d4e5.webp"], {"Фото.webp": ["different"]})
        self.assertEqual(candidate["status"], "candidate")
        self.assertFalse(candidate["confirmed"])
        verified = catalog.decide_join(
            {"Название фото": "photo.webp", "Slug": "photo"}, ["uploads/photo_a1b2c3d4e5.webp"], rows,
            {"member": "uploads/photo_a1b2c3d4e5.webp", "evidence": ["label reads PHOTO"]})
        self.assertTrue(verified["confirmed"])
        self.assertEqual(verified["method"], "manual_verified")
        self.assertIn("label reads PHOTO", verified["evidence"])

    def test_shared_filename_is_ambiguous(self):
        row = {"Название фото": "same.webp", "Slug": "vintage-2024"}
        join = catalog.decide_join(row, ["uploads/same_a1b2c3d4e5.webp"],
                                   {"same.webp": ["vintage-2024", "vintage-2025"]})
        self.assertEqual(join["status"], "ambiguous")

    def test_published_image_path_resolves_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            old_root, old_project = catalog.ROOT, catalog.PROJECT
            try:
                catalog.ROOT = Path(directory)
                catalog.PROJECT = catalog.ROOT / "outputs/wine-ml-api"
                source = catalog.ROOT / "work/wine-ml/catalog_images/hash.webp"
                source.parent.mkdir(parents=True)
                source.write_bytes(b"image")
                image = catalog.publish_image({"path": "work/wine-ml/catalog_images/hash.webp"})
                self.assertEqual(image["path"], "data/catalog_images/hash.webp")
                self.assertTrue((catalog.PROJECT / image["path"]).is_file())
            finally:
                catalog.ROOT, catalog.PROJECT = old_root, old_project

    def test_csv_deduplicates_and_rejects_conflicting_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            fields = ["Название вина", "Категория", "Цвет", "Регион", "Сорт винограда",
                      "Описание", "Винодельня", "Slug", "Название фото"]
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                row = dict.fromkeys(fields, "x")
                row["Slug"] = "slug"
                writer.writerows([row, row])
            rows, source, raw = catalog.read_catalog(path)
            self.assertEqual((len(rows), raw, source["slug"]), (1, 2, [2, 3]))


if __name__ == "__main__":
    unittest.main()
