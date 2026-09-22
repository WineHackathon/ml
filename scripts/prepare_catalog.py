#!/usr/bin/env python3
"""Build a conservative, reproducible wine catalog from CSV and Strapi uploads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
DEFAULT_WORK = ROOT / "work/catalog"
IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff", ".heic"}
THUMBNAIL_PREFIXES = ("thumbnail_", "small_", "medium_", "large_")
TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
        "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)
HASH_SUFFIX = re.compile(r"_[0-9a-f]{10}$", re.IGNORECASE)


def normalize_name(value: str) -> str:
    value = HASH_SUFFIX.sub("", Path(value).stem).lower().translate(TRANSLIT)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def unhashed_name(value: str) -> str:
    path = Path(value)
    return HASH_SUFFIX.sub("", path.stem) + path.suffix.lower()


def is_safe_member(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", value)


def read_catalog(path: Path) -> tuple[list[dict[str, str]], dict[str, list[int]], int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows_by_slug: dict[str, dict[str, str]] = {}
    source_rows: dict[str, list[int]] = defaultdict(list)
    for line, row in enumerate(raw, 2):
        row = {key: (value or "").strip() for key, value in row.items()}
        slug = row["Slug"]
        source_rows[slug].append(line)
        previous = rows_by_slug.setdefault(slug, row)
        if previous != row:
            raise ValueError(f"conflicting rows for slug {slug!r}: lines {source_rows[slug]}")
    return list(rows_by_slug.values()), dict(source_rows), len(raw)


def run(command: list[str], *, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in ok:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-4000:]}")
    return result


def archive_members(unrar: Path, archive: Path) -> list[str]:
    result = run([str(unrar), "lb", "-v", "-c-", "-cfg-", str(archive)])
    members = list(dict.fromkeys(line for line in result.stdout.splitlines() if line.strip()))
    unsafe = [member for member in members if not is_safe_member(member)]
    if unsafe:
        raise ValueError(f"unsafe archive members: {unsafe[:5]}")
    return members


def classify_member(member: str) -> str:
    name = Path(member).name.lower()
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        return "cms_non_image"
    if name.startswith(THUMBNAIL_PREFIXES):
        return "cms_derivative"
    return "original_candidate"


def decide_join(row: dict[str, str], candidates: list[str], photo_slugs: dict[str, list[str]],
                manual_verification: dict | str | None = None) -> dict:
    photo = row["Название фото"]
    slug = row["Slug"]
    evidence = [f"csv_photo_normalized={normalize_name(photo)}"]
    manual_member = (manual_verification.get("member") if isinstance(manual_verification, dict)
                     else manual_verification)
    if manual_member:
        if manual_member not in candidates:
            raise ValueError(f"manual verification for {slug!r} is not one of its candidates")
        review_evidence = (manual_verification.get("evidence", [])
                           if isinstance(manual_verification, dict) else [])
        return {"method": "manual_verified", "confidence": "confirmed", "confirmed": True,
                "status": "ready", "member": manual_member, "candidates": candidates,
                "evidence": evidence + ["explicit slug-to-member verification file"] + review_evidence}
    if len(photo_slugs[photo]) > 1:
        return {"method": "filename_collision", "confidence": "none", "confirmed": False,
                "status": "ambiguous", "member": None, "candidates": candidates,
                "evidence": evidence + [f"shared_by_slugs={','.join(photo_slugs[photo])}"]}
    if not candidates:
        return {"method": "no_candidate", "confidence": "none", "confirmed": False,
                "status": "missing", "member": None, "candidates": [], "evidence": evidence}

    slug_matches = [item for item in candidates if normalize_name(Path(item).name) == normalize_name(slug)]
    if len(slug_matches) == 1:
        member = slug_matches[0]
        return {"method": "dual_key_candidate", "confidence": "candidate", "confirmed": False,
                "status": "candidate", "member": member, "candidates": candidates,
                "evidence": evidence + ["archive name also matches normalized slug; no independent asset ID"]}

    exact = [item for item in candidates if unhashed_name(Path(item).name).casefold() == photo.casefold()]
    if len(candidates) == 1 and len(exact) == 1:
        return {"method": "exact_filename_candidate", "confidence": "candidate", "confirmed": False,
                "status": "candidate", "member": exact[0], "candidates": candidates,
                "evidence": evidence + ["archive name equals CSV filename; filename is not an asset ID"]}
    if len(candidates) == 1:
        return {"method": "normalized_singleton", "confidence": "candidate", "confirmed": False,
                "status": "candidate", "member": candidates[0], "candidates": candidates,
                "evidence": evidence + ["normalization alone is not identity proof"]}
    return {"method": "multiple_candidates", "confidence": "none", "confirmed": False,
            "status": "ambiguous", "member": None, "candidates": candidates, "evidence": evidence}


def extract_members(unrar: Path, archive: Path, members: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    listfile = destination.parent / "selected_members.txt"
    listfile.write_text("\n".join(members) + "\n", encoding="utf-8")
    # Safe members were checked above; `x` preserves identity when basenames collide and `-ol-` refuses links.
    run([str(unrar), "x", "-idq", "-o+", "-ol-", "-cfg-", str(archive),
         f"@{listfile}", str(destination) + os.sep])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    result = run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "format", str(path)])
    values = dict(re.findall(r"^\s*(pixelWidth|pixelHeight|format):\s*(.+)$", result.stdout, re.MULTILINE))
    if not {"pixelWidth", "pixelHeight", "format"} <= values.keys():
        raise ValueError(f"cannot decode image: {path}")
    return {"width": int(values["pixelWidth"]), "height": int(values["pixelHeight"]),
            "format": values["format"].lower()}


def eval_overlaps(eval_zip: Path | None, real_photos: Path | None) -> dict:
    result = {"labels_inferred": False, "queries": []}
    real_by_hash: dict[str, list[str]] = defaultdict(list)
    if real_photos and real_photos.exists():
        for path in sorted(item for item in real_photos.iterdir() if item.is_file()):
            real_by_hash[sha256(path)].append(path.name)
    if not eval_zip or not eval_zip.exists():
        return result
    with zipfile.ZipFile(eval_zip) as archive:
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if (info.is_dir() or "__MACOSX" in parts or Path(info.filename).name.startswith("._")
                    or Path(info.filename).suffix.lower() not in IMAGE_EXTENSIONS):
                continue
            digest = hashlib.sha256(archive.read(info)).hexdigest()
            result["queries"].append({"entry": info.filename, "sha256": digest,
                                      "real_photo_matches": real_by_hash.get(digest, [])})
    result["overlap_count"] = sum(bool(item["real_photo_matches"]) for item in result["queries"])
    return result


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def publish_image(image: dict) -> dict:
    source = ROOT / image["path"]
    target = PROJECT / "data/catalog_images" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    return {**image, "path": target.relative_to(PROJECT).as_posix()}


def build(args: argparse.Namespace) -> dict:
    work = args.work_dir.resolve()
    if ROOT not in work.parents and work != ROOT:
        raise ValueError("--work-dir must remain inside the workspace")
    work.mkdir(parents=True, exist_ok=True)
    rows, source_rows, raw_count = read_catalog(args.csv)
    members = archive_members(args.unrar, args.archive)
    if args.test_archive:
        run([str(args.unrar), "t", "-idq", "-cfg-", str(args.archive)])

    inventory = [{"member": member, "kind": classify_member(member)} for member in members]
    (work / "archive_inventory.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in inventory), encoding="utf-8")
    originals = [item["member"] for item in inventory if item["kind"] == "original_candidate"]
    by_normalized: dict[str, list[str]] = defaultdict(list)
    for member in originals:
        by_normalized[normalize_name(Path(member).name)].append(member)
    photo_slugs: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        photo_slugs[row["Название фото"]].append(row["Slug"])
    manual = json.loads(args.manual_verifications.read_text(encoding="utf-8")) if args.manual_verifications.exists() else {}
    joins = {row["Slug"]: decide_join(row, sorted(by_normalized[normalize_name(row["Название фото"])]),
                                      photo_slugs, manual.get(row["Slug"]))
             for row in rows}

    selected = sorted({join["member"] for join in joins.values() if join["member"]})
    extracted = work / "extracted"
    extract_members(args.unrar, args.archive, selected, extracted)
    catalog_images = work / "catalog_images"
    catalog_images.mkdir(exist_ok=True)
    image_metadata: dict[str, dict] = {}
    invalid_members: dict[str, str] = {}
    for member in selected:
        source = extracted / member
        try:
            digest = sha256(source)
            metadata = inspect_image(source)
            target = catalog_images / f"{digest}{source.suffix.lower()}"
            if not target.exists():
                os.link(source, target)
            image_metadata[member] = {"path": relative_to_root(target), "sha256": digest, **metadata}
        except (OSError, RuntimeError, ValueError) as error:
            invalid_members[member] = str(error)

    manifest = []
    issues = []
    asset_slugs: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        join = joins[row["Slug"]]
        if join["member"]:
            asset_slugs[join["member"]].append(row["Slug"])
    for row in sorted(rows, key=lambda item: item["Slug"]):
        slug = row["Slug"]
        join = joins[slug]
        image = image_metadata.get(join["member"])
        status = join["status"]
        if join["member"] in invalid_members:
            status, image = "invalid", None
            join = {**join, "confirmed": False, "confidence": "none",
                    "evidence": join["evidence"] + [invalid_members[join["member"]]]}
        if status in {"ready", "candidate"} and image:
            image = publish_image(image)
        reasons = [] if status == "ready" else [status]
        item = {
            "wine_id": slug,
            "canonical_name": row["Название вина"] or None,
            "producer": row["Винодельня"] or None,
            "region": row["Регион"] or None,
            "color": row["Цвет"] or None,
            "type": row["Категория"] or None,
            "grapes": row["Сорт винограда"] or None,
            "description": row["Описание"] or None,
            "image": image,
            "source": {"csv": str(args.csv), "rows": source_rows[slug],
                       "archive": str(args.archive), "member": join["member"]},
            "aliases": {"filenames": [row["Название фото"]],
                        "slugs": asset_slugs.get(join["member"], [slug])},
            "join": {key: value for key, value in join.items() if key not in {"status", "member"}},
            "status": {"catalog_image": status,
                       "training_eligible": status == "ready" and join["confirmed"], "reasons": reasons},
        }
        manifest.append(item)
        if status != "ready":
            issues.append({"wine_id": slug, "status": status, "photo": row["Название фото"],
                           "method": join["method"], "candidates": join["candidates"]})

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in manifest),
        encoding="utf-8")
    args.issues.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in issues), encoding="utf-8")
    overlap = eval_overlaps(args.eval_zip, args.real_photos)
    (work / "eval_overlap.json").write_text(json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status_counts = Counter(item["status"]["catalog_image"] for item in manifest)
    method_counts = Counter(item["join"]["method"] for item in manifest)
    collision_names = {name: slugs for name, slugs in photo_slugs.items() if len(slugs) > 1}
    volume_pattern = re.sub(r"part\d+\.rar$", "part*.rar", args.archive.name, flags=re.IGNORECASE)
    volumes = sorted(args.archive.parent.glob(volume_pattern)) or [args.archive]
    unrar_version = next((line for line in run([str(args.unrar)]).stdout.splitlines() if line.strip()), "unknown")
    report = {
        "complete": status_counts["ready"] == len(manifest),
        "inputs": {"csv": {"path": str(args.csv), "sha256": sha256(args.csv)},
                   "archive_volumes": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                                       for path in volumes],
                   "sources_modified": False},
        "tools": {"unrar": unrar_version, "image_decoder": "macOS sips"},
        "csv": {"raw_rows": raw_count, "unique_slugs": len(rows), "duplicates_removed": raw_count - len(rows)},
        "archive": {"members": len(members), "tested": bool(args.test_archive),
                    "kinds": dict(Counter(item["kind"] for item in inventory)),
                    "selected_assets": len(selected), "decoded_assets": len(image_metadata),
                    "unique_decoded_sha256": len({item["sha256"] for item in image_metadata.values()}),
                    "invalid_assets": invalid_members},
        "catalog": {"rows": len(manifest), "status": dict(status_counts), "join_methods": dict(method_counts),
                    "confirmed_coverage": status_counts["ready"] / len(manifest),
                    "provisional_coverage": (status_counts["ready"] + status_counts["candidate"]) / len(manifest)},
        "collisions": {"filename_count": len(collision_names),
                       "affected_slug_count": sum(len(value) for value in collision_names.values()),
                       "filenames": collision_names},
        "eval_overlap": overlap,
        "policy": {"confirmed": ["manual_verified", "strapi_id"],
                   "candidate": "filename, dual-key, and normalized singleton matches are not identity proof",
                   "ambiguous": "shared CSV filenames and multiple archive candidates are never auto-merged"},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "catalog_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--csv", type=Path, required=True)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--unrar", type=Path, default=Path("unrar"))
    result.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    result.add_argument("--output-manifest", type=Path,
                        default=ROOT / "data/catalog_manifest.jsonl")
    result.add_argument("--issues", type=Path, default=ROOT / "work/catalog/issues.jsonl")
    result.add_argument("--report", type=Path, default=ROOT / "work/catalog/report.json")
    result.add_argument("--eval-zip", type=Path, default=ROOT / "work/catalog/no-eval.zip")
    result.add_argument("--real-photos", type=Path, default=ROOT / "work/catalog/no-real-photos")
    result.add_argument("--manual-verifications", type=Path,
                        default=ROOT / "work/catalog/manual_verifications.json")
    result.add_argument("--no-test-archive", action="store_false", dest="test_archive")
    result.set_defaults(test_archive=True)
    return result


if __name__ == "__main__":
    built = build(parser().parse_args())
    print(json.dumps(built, ensure_ascii=False, indent=2))
