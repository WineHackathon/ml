#!/usr/bin/env python3
"""Apply verified public-card asset repairs and run a frozen-index A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wine_api.preprocess import decode_image, retrieval_views  # noqa: E402
from wine_api.retrieval import (  # noqa: E402
    CatalogIndex,
    DEFAULT_REVISION,
    ImageEncoder,
    build_index,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unrar_member(unrar: Path, archive: Path, member: str) -> bytes:
    return subprocess.run(
        [str(unrar), "p", "-inul", str(archive), member], check=True, capture_output=True
    ).stdout


def apply_patches(
    manifest: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    asset_dir: Path,
    read_asset: Callable[[str], bytes],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = copy.deepcopy(manifest)
    by_slug = {row["wine_id"]: row for row in repaired}
    if len(by_slug) != len(repaired):
        raise ValueError("manifest wine_id values must be unique")
    patch_slugs = [patch.get("slug") for patch in patches]
    if len(set(patch_slugs)) != len(patch_slugs):
        raise ValueError("patch slugs must be unique")

    receipts = []
    asset_dir.mkdir(parents=True, exist_ok=True)
    for patch in patches:
        slug = patch.get("slug")
        matches = patch.get("archive_matches")
        patch_status = patch.get("status")
        if (
            patch_status not in {"resolved_archive_exact_asset", "resolved_official_public_asset"}
            or patch.get("provenance") != "official_public_card_ssr_and_og_image"
            or patch.get("source_url") != f"https://vino-svoe.ru/wines/{slug}"
            or patch.get("final_url") != patch.get("source_url")
            or not patch.get("raw_html_sha256")
            or sha256(Path(patch.get("raw_html_path", "")).read_bytes()) != patch["raw_html_sha256"]
        ):
            raise ValueError(f"unverified or ambiguous patch: {slug}")
        if patch_status == "resolved_archive_exact_asset" and (
            not isinstance(matches, list) or len(matches) != 1
            or Path(matches[0]).name != patch.get("asset")
        ):
            raise ValueError(f"unverified or ambiguous patch: {slug}")
        record = by_slug.get(slug)
        if not record or record.get("status", {}).get("catalog_image") not in {"missing", "ambiguous"}:
            raise ValueError(f"patch target is absent or not unresolved: {slug}")
        status_before = record["status"]["catalog_image"]

        data = (read_asset(matches[0]) if patch_status == "resolved_archive_exact_asset"
                else Path(patch["asset_path"]).read_bytes())
        image = decode_image(data)
        digest = sha256(data)
        if patch_status == "resolved_official_public_asset" and (
            digest != patch.get("asset_sha256")
            or [image.width, image.height] != patch.get("asset_pixels")
        ):
            raise ValueError(f"downloaded public asset changed: {slug}")
        suffix = (image.format or "webp").lower()
        output = asset_dir / f"{digest}.{suffix}"
        if not output.exists():
            output.write_bytes(data)
        record["image"] = {
            "path": output.resolve().as_posix(),
            "sha256": digest,
            "width": image.width,
            "height": image.height,
            "format": suffix,
        }
        record["join"] = {
            "method": ("official_public_card_exact_archive_asset"
                       if patch_status == "resolved_archive_exact_asset"
                       else "official_public_card_asset"),
            "confidence": "confirmed",
            "confirmed": True,
            "candidates": matches or [],
            "evidence": [patch["source_url"], patch["image_url"],
                         matches[0] if matches else patch["asset_path"]],
        }
        record["status"] = {"catalog_image": "ready", "training_eligible": True, "reasons": []}
        record["asset_repair"] = {
            "source_url": patch["source_url"],
            "image_url": patch["image_url"],
            "archive_member": matches[0] if matches else None,
            "provenance": patch["provenance"],
            "raw_html_sha256": patch["raw_html_sha256"],
            "fetched_at": patch.get("fetched_at"),
        }
        receipts.append({
            "slug": slug,
            "source_status_before": status_before,
            "sha256": digest,
            "pixels": [image.width, image.height],
            "format": suffix,
            "archive_member": matches[0] if matches else None,
            "source_url": patch["source_url"],
            "source_kind": patch_status,
        })
        image.close()
    return repaired, receipts


def build_source_report(original: list[dict[str, Any]], repaired: list[dict[str, Any]],
                        patches: list[dict[str, Any]], receipts: list[dict[str, Any]],
                        manifest_path: Path, patches_path: Path) -> dict[str, Any]:
    final_counts = Counter(row["status"]["catalog_image"] for row in repaired)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for receipt in receipts:
        by_hash[receipt["sha256"]].append(receipt["slug"])
    accounting = {
        "confirmed_source": final_counts["ready"],
        "provisional": final_counts["candidate"],
        "missing": final_counts["missing"],
        "ambiguous": final_counts["ambiguous"],
    }
    return {
        "kind": "catalog_v2_public_source_recovery",
        "claims": {"official_hidden_accuracy": None, "all_manifest_rows_accounted": sum(accounting.values()) == len(repaired)},
        "counts": {
            "manifest_rows": len(repaired),
            "audited_unresolved_rows": len(patches),
            "applied_repairs": len(receipts),
            "baseline_catalog_status": dict(Counter(row["status"]["catalog_image"] for row in original)),
            "patch_status": dict(Counter(row["status"] for row in patches)),
            "final_catalog_status": dict(final_counts),
            "accounting": accounting,
        },
        "identical_asset_groups": [
            {"sha256": digest, "slugs": slugs, "rule": "shared bytes only; target slugs remain distinct"}
            for digest, slugs in sorted(by_hash.items()) if len(slugs) > 1
        ],
        "inputs": {"source_patches_sha256": sha256(patches_path.read_bytes())},
        "outputs": {"catalog_manifest": manifest_path.resolve().as_posix(),
                    "catalog_manifest_sha256": sha256(manifest_path.read_bytes())},
        "repair_receipts": receipts,
    }


def metric(rows: list[dict[str, Any]], arm: str, k: int) -> int:
    return sum(row["true_slug"] in row[f"{arm}_top5"][:k] for row in rows)


def evaluate(base_path: Path, repaired_path: Path, annotations_path: Path, device: str) -> dict[str, Any]:
    base, repaired = CatalogIndex.load(base_path), CatalogIndex.load(repaired_path)
    for key in ("model_source", "model_revision", "gallery_policy"):
        if base.metadata[key] != repaired.metadata[key]:
            raise ValueError(f"index mismatch: {key}")
    encoder = ImageEncoder(base.metadata["model_source"], base.metadata["model_revision"], device, True)
    base_slugs = {entry["slug"] for entry in base.entries}
    repaired_slugs = {entry["slug"] for entry in repaired.entries}
    old_rows = [i for i, entry in enumerate(repaired.entries) if entry["slug"] in base_slugs]
    control = CatalogIndex(repaired.embeddings[old_rows], [repaired.entries[i] for i in old_rows], dict(repaired.metadata))
    repaired_positions = {(entry["slug"], entry["view"]): i for i, entry in enumerate(repaired.entries)}
    unchanged = repaired.embeddings[[repaired_positions[(entry["slug"], entry["view"])] for entry in base.entries]]
    unchanged_cosines = (base.embeddings * unchanged).sum(axis=1)
    rows, encode_times, base_times, repaired_times = [], [], [], []
    root = annotations_path.parent.parent
    for label in load_jsonl(annotations_path):
        path = Path(label["image_path"])
        path = path if path.is_absolute() else root / path
        image = decode_image(path.read_bytes())
        views = retrieval_views(image)
        started = time.perf_counter()
        query = encoder.embed([view for _, view in views])
        encode_times.append(time.perf_counter() - started)
        for _, view in views:
            view.close()
        started = time.perf_counter()
        baseline = base.search(query, 5)
        base_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        patched = repaired.search(query, 5)
        repaired_times.append(time.perf_counter() - started)
        controlled = control.search(query, 5)
        true_slug = label.get("true_slug")
        rows.append({
            "query_id": label["query_id"],
            "label_status": label.get("label_status"),
            "true_slug": true_slug,
            "base_in_gallery": bool(true_slug and true_slug in base_slugs),
            "repaired_in_gallery": bool(true_slug and true_slug in repaired_slugs),
            "baseline_top5": [item["slug"] for item in baseline],
            "repaired_top5": [item["slug"] for item in patched],
            "old_gallery_control_top5": [item["slug"] for item in controlled],
            "baseline_rank": next((i for i, item in enumerate(baseline, 1) if item["slug"] == true_slug), None),
            "repaired_rank": next((i for i, item in enumerate(patched, 1) if item["slug"] == true_slug), None),
            "top1_changed": baseline[0]["slug"] != patched[0]["slug"],
        })
    exact = [row for row in rows if row["true_slug"]]
    common = [row for row in exact if row["base_in_gallery"] and row["repaired_in_gallery"]]
    repaired_covered = [row for row in exact if row["repaired_in_gallery"]]
    return {
        "kind": "catalog_asset_repair_silver_proxy_ab",
        "accuracy_scope": "exploratory_silver_only",
        "labels_used_for_repair_or_tuning": False,
        "aggregation": "same frozen encoder and query views; max similarity across views then unique slug",
        "inputs": {
            "annotations_sha256": sha256(annotations_path.read_bytes()),
            "base_index_sha256": sha256(base_path.read_bytes()),
            "repaired_index_sha256": sha256(repaired_path.read_bytes()),
            "model_source": base.metadata["model_source"],
            "model_revision": base.metadata["model_revision"],
        },
        "unchanged_embedding_consistency": {
            "rows": len(base.entries),
            "cosine_min": float(unchanged_cosines.min()),
            "cosine_median": float(statistics.median(unchanged_cosines)),
            "max_abs_delta": float(abs(base.embeddings - unchanged).max()),
        },
        "counts": {
            "annotations": len(rows),
            "overall_exact": len(exact),
            "base_gallery_exact": sum(row["base_in_gallery"] for row in exact),
            "repaired_gallery_exact": len(repaired_covered),
            "common_gallery_exact": len(common),
        },
        "newly_covered_exact_targets": [row["true_slug"] for row in exact if not row["base_in_gallery"] and row["repaired_in_gallery"]],
        "metrics": {
            "overall_exact": {
                "denominator": len(exact),
                "baseline_top1": metric(exact, "baseline", 1),
                "repaired_top1": metric(exact, "repaired", 1),
                "baseline_top5": metric(exact, "baseline", 5),
                "repaired_top5": metric(exact, "repaired", 5),
            },
            "common_gallery_exact": {
                "denominator": len(common),
                "baseline_top1": metric(common, "baseline", 1),
                "repaired_top1": metric(common, "repaired", 1),
                "baseline_top5": metric(common, "baseline", 5),
                "repaired_top5": metric(common, "repaired", 5),
                "old_gallery_control_top1": metric(common, "old_gallery_control", 1),
                "old_gallery_control_top5": metric(common, "old_gallery_control", 5),
            },
            "repaired_gallery_exact": {
                "denominator": len(repaired_covered),
                "top1": metric(repaired_covered, "repaired", 1),
                "top5": metric(repaired_covered, "repaired", 5),
            },
        },
        "timings_seconds": {
            "query_encode_median": statistics.median(encode_times),
            "baseline_search_median": statistics.median(base_times),
            "repaired_search_median": statistics.median(repaired_times),
            "evaluation_total": sum(encode_times) + sum(base_times) + sum(repaired_times),
        },
        "changed_query_ids": [row["query_id"] for row in rows if row["top1_changed"]],
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/catalog_manifest.jsonl"))
    parser.add_argument("--patches", type=Path, default=Path("artifacts/catalog_asset_patches.jsonl"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--unrar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/catalog_repair"))
    parser.add_argument("--base-index", type=Path, default=Path("artifacts/catalog_index.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/proxy_annotations.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest-only", action="store_true",
                        help="write the versioned manifest/report without building an index")
    args = parser.parse_args()

    output_manifest = args.output_dir / "catalog_manifest.jsonl"
    output_index = args.output_dir / "catalog_index.npz"
    manifest = load_jsonl(args.manifest)
    source_patches = load_jsonl(args.patches)
    resolved = [patch for patch in source_patches if patch.get("status") in {
        "resolved_archive_exact_asset", "resolved_official_public_asset"
    }]
    repaired, receipts = apply_patches(
        manifest, resolved, args.output_dir / "assets",
        lambda member: unrar_member(args.unrar, args.archive, member),
    )
    source_root = args.manifest.parent.parent.resolve()
    for record in repaired:
        image = record.get("image")
        if image and not Path(image["path"]).is_absolute():
            image["path"] = (source_root / image["path"]).as_posix()
    output_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired), encoding="utf-8")
    report = build_source_report(manifest, repaired, source_patches, receipts,
                                 output_manifest, args.patches)
    report_path = args.output_dir / "source_patch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.manifest_only:
        print(json.dumps(report["counts"], ensure_ascii=False))
        return
    base = CatalogIndex.load(args.base_index)
    build = build_index(
        output_manifest, output_index, base.metadata["model_source"],
        base.metadata.get("model_revision") or DEFAULT_REVISION, args.device, 8, True, True,
    )
    if build["indexed_slugs"] != base.metadata["indexed_slugs"] + len(receipts):
        raise RuntimeError("repaired gallery did not preserve the base gallery and add every patch")
    evaluation = evaluate(args.base_index, output_index, args.annotations, args.device)
    evaluation["repair_receipts"] = receipts
    evaluation["repaired_index_metadata"] = build
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": evaluation["counts"], "metrics": evaluation["metrics"], "timings_seconds": evaluation["timings_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
