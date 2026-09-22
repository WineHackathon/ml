import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "recover_catalog_routes", Path(__file__).parents[1] / "scripts/recover_catalog_routes.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_photo_match_ignores_only_strapi_hash_and_separators():
    assert MODULE.photo_matches("DSC00839-Photoroom.webp",
                                "DSC_00839_Photoroom_418c4b31b3.webp")
    assert not MODULE.photo_matches("DSC00836.webp", "DSC_00839_Photoroom_418c4b31b3.webp")
