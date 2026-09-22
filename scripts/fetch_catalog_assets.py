#!/usr/bin/env python3
"""Fetch public card evidence without changing live catalog data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter

from PIL import Image


BASE = "https://vino-svoe.ru"
USER_AGENT = "wine-catalog-audit/1.0 (+bounded public metadata check)"
DEFAULT_INVENTORY = Path(__file__).resolve().parents[3] / "work/wine-ml/archive_inventory.jsonl"
BLOCKING_HTTP_CODES = {401, 403, 429}


class AccessBlocked(RuntimeError):
    pass


class MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.nuxt_chunks: list[str] = []
        self._nuxt = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("property") in {"og:title", "og:image"}:
            self.meta[values["property"]] = values.get("content", "")
        self._nuxt = tag == "script" and values.get("id") == "__NUXT_DATA__"

    def handle_data(self, data: str) -> None:
        if self._nuxt:
            self.nuxt_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._nuxt = False


def parse_card(html: str, slug: str) -> dict[str, str]:
    parser = MetaParser()
    parser.feed(html)
    image_url = parser.meta.get("og:image", "")
    image = urllib.parse.urlparse(image_url)
    nuxt = "".join(parser.nuxt_chunks)
    if (not parser.meta.get("og:title") or image.scheme != "https"
            or image.netloc != "api.vino-svoe.ru" or "/uploads/" not in image.path
            or f'"wine-{slug}"' not in nuxt or f'"{slug}"' not in nuxt):
        raise ValueError("page does not contain a matching official wine record")
    return {"slug": slug, "name": parser.meta["og:title"].strip(),
            "image_url": image_url, "asset": Path(image.path).name}


def selected_rows(path: Path, statuses: set[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("status", {}).get("catalog_image") in statuses]


def unresolved(path: Path) -> list[dict]:
    return selected_rows(path, {"missing", "ambiguous"})


def archive_assets(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "original_candidate":
            result.setdefault(Path(row["member"]).name, []).append(row["member"])
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(path)


def verify_image(data: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        return (image.format or "webp").lower(), image.width, image.height


def matched_asset_status(manifest_status: str, asset: str,
                         manifest_candidates: list[str]) -> str:
    if manifest_status != "candidate":
        return "resolved_archive_exact_asset"
    return ("corroborated_manifest_asset"
            if asset in {Path(item).name for item in manifest_candidates}
            else "changed_official_archive_asset")


def candidate_audit_report(manifest: Path, patches: list[dict], sample_seed: str | None) -> dict:
    counts = Counter(row["status"] for row in patches)
    changed = counts["changed_official_archive_asset"] + counts["changed_official_public_asset"]
    fetched = [row["fetched_at"] for row in patches if row.get("fetched_at")]
    total = len(selected_rows(manifest, {"candidate"}))
    return {
        "kind": "catalog_candidate_official_card_sample_audit" if len(patches) < total
                else "catalog_candidate_official_card_full_audit",
        "scope": {"candidate_rows": total, "audited_rows": len(patches),
                  "sample_seed": sample_seed, "sample_is_proof_of_remaining_rows": False},
        "counts": {"status": dict(counts), "concordant": counts["corroborated_manifest_asset"],
                   "changed": changed, "page_not_found": counts["page_not_found"]},
        "rates": {key: value / len(patches) for key, value in {
            "concordant": counts["corroborated_manifest_asset"],
            "changed": changed,
            "page_not_found": counts["page_not_found"],
        }.items()},
        "decision": ("continue_full_audit_due_to_changed_assets" if changed
                     else "stop_after_sample_no_changed_asset_signal"),
        "source_version_drift": {
            "baseline_manifest_sha256": sha256(manifest.read_bytes()),
            "current_public_site_fetch_window_utc": [min(fetched), max(fetched)] if fetched else None,
            "meaning": "changed assets reflect current official cards versus the supplied archive-era candidate",
        },
    }


class Fetcher:
    def __init__(self, timeout: float, interval: float = 1.0, retries: int = 2) -> None:
        self.timeout, self.interval, self.retries, self.last_request = timeout, interval, retries, 0.0

    def get(self, url: str) -> tuple[bytes, str]:
        for attempt in range(self.retries + 1):
            wait = self.interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read(), response.geturl()
            except urllib.error.HTTPError as error:
                if error.code in BLOCKING_HTTP_CODES:
                    raise AccessBlocked(f"HTTP {error.code} for {url}") from error
                if error.code < 500 or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise
            finally:
                self.last_request = time.monotonic()
        raise RuntimeError("unreachable")


class RobotsRules:
    def __init__(self, text: str) -> None:
        self.rules: list[tuple[bool, str]] = []
        active = False
        saw_rules = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            field, value = (part.strip() for part in line.split(":", 1))
            if field.lower() == "user-agent":
                if saw_rules:
                    active, saw_rules = False, False
                active = value == "*"
            elif field.lower() in {"allow", "disallow"} and active:
                self.rules.append((field.lower() == "allow", value))
                saw_rules = True

    def can_fetch(self, _user_agent: str, url: str) -> bool:
        target = urllib.parse.urlsplit(url).path
        if urllib.parse.urlsplit(url).query:
            target += "?" + urllib.parse.urlsplit(url).query
        matches = []
        for allowed, pattern in self.rules:
            if not pattern:
                continue
            end = pattern.endswith("$")
            escaped = re.escape(pattern[:-1] if end else pattern).replace(r"\*", ".*")
            if re.match("^" + escaped + ("$" if end else ""), target):
                matches.append((len(pattern.replace("*", "")), allowed))
        return max(matches, default=(0, True))[1]


def load_robots(fetcher: Fetcher, base: str) -> RobotsRules:
    url = f"{base}/robots.txt"
    data, final_url = fetcher.get(url)
    if final_url != url:
        raise ValueError(f"unexpected robots redirect: {final_url}")
    return RobotsRules(data.decode("utf-8"))


def run(manifest: Path, output: Path, limit: int, timeout: float,
        inventory: Path = DEFAULT_INVENTORY, slugs: list[str] | None = None,
        raw_dir: Path | None = None, asset_dir: Path | None = None,
        resume: bool = True, retry_errors: bool = False,
        statuses: set[str] | None = None, sample_seed: str | None = None,
        resume_from: Path | None = None) -> list[dict]:
    statuses = statuses or {"missing", "ambiguous"}
    rows = selected_rows(manifest, statuses)
    if sample_seed:
        rows.sort(key=lambda row: sha256(f"{sample_seed}:{row['wine_id']}".encode()))
    if not 1 <= limit <= len(rows):
        raise ValueError(f"--limit must be between 1 and {len(rows)}")
    if slugs:
        requested = set(slugs)
        rows = [row for row in rows if row["wine_id"] in requested]
    rows = rows[:limit]
    raw_dir = raw_dir or output.parent / "raw_cards"
    asset_dir = asset_dir or output.parent / "assets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    wanted = {item["wine_id"] for item in rows}
    resume_path = output if output.exists() else resume_from
    previous = ([] if not resume or not resume_path or not resume_path.exists()
                else [json.loads(line) for line in resume_path.read_text(encoding="utf-8").splitlines() if line])
    results = {row["slug"]: row for row in previous if row.get("slug") in wanted}
    if retry_errors:
        results = {slug: row for slug, row in results.items()
                   if row.get("status") != "fetch_or_parse_error"}
    fetcher = Fetcher(timeout)
    page_robots = load_robots(fetcher, BASE)
    image_robots = load_robots(fetcher, "https://api.vino-svoe.ru")
    assets = archive_assets(inventory)
    blocked = None

    for manifest_row in rows:
        slug = manifest_row["wine_id"]
        if slug in results:
            continue
        url = f"{BASE}/wines/{urllib.parse.quote(slug)}"
        if blocked:
            results[slug] = {"slug": slug, "manifest_status": manifest_row["status"]["catalog_image"],
                             "status": "not_attempted_blocked", "source_url": url,
                             "error": blocked}
            continue
        if not page_robots.can_fetch(USER_AGENT, url):
            results[slug] = {"slug": slug, "manifest_status": manifest_row["status"]["catalog_image"],
                             "status": "robots_disallowed", "source_url": url}
            continue
        try:
            html_bytes, final_url = fetcher.get(url)
            expected = urllib.parse.urlparse(url).path.rstrip("/")
            if urllib.parse.urlparse(final_url).path.rstrip("/") != expected:
                raise ValueError("unexpected redirect")
            html = html_bytes.decode("utf-8")
            card = parse_card(html, slug)
            raw_digest = sha256(html_bytes)
            raw_path = raw_dir / f"{raw_digest}.html"
            if not raw_path.exists():
                raw_path.write_bytes(html_bytes)
            matches = assets.get(card["asset"], [])
            result = {
                **card,
                "manifest_name": manifest_row.get("canonical_name"),
                "manifest_status": manifest_row["status"]["catalog_image"],
                "manifest_candidates": manifest_row.get("join", {}).get("candidates", []),
                "manifest_image_sha256": (manifest_row.get("image") or {}).get("sha256"),
                "source_url": url,
                "final_url": final_url,
                "archive_matches": matches,
                "raw_html_sha256": raw_digest,
                "raw_html_path": raw_path.resolve().as_posix(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "provenance": "official_public_card_ssr_and_og_image",
            }
            if len(matches) == 1:
                result["status"] = matched_asset_status(
                    manifest_row["status"]["catalog_image"], card["asset"], result["manifest_candidates"])
            elif len(matches) > 1:
                result["status"] = "ambiguous_archive_exact_asset"
            elif not image_robots.can_fetch(USER_AGENT, card["image_url"]):
                result["status"] = "image_robots_disallowed"
            else:
                image_bytes, image_final_url = fetcher.get(card["image_url"])
                if image_final_url != card["image_url"]:
                    raise ValueError("unexpected image redirect")
                image_format, width, height = verify_image(image_bytes)
                digest = sha256(image_bytes)
                image_path = asset_dir / f"{digest}.{image_format}"
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                result.update({
                    "status": ("changed_official_public_asset"
                               if manifest_row["status"]["catalog_image"] == "candidate"
                               else "resolved_official_public_asset"),
                    "asset_path": image_path.resolve().as_posix(),
                    "asset_sha256": digest,
                    "asset_pixels": [width, height],
                    "asset_format": image_format,
                })
            results[slug] = result
        except AccessBlocked as error:
            blocked = f"{type(error).__name__}: {error}"
            results[slug] = {"slug": slug, "manifest_status": manifest_row["status"]["catalog_image"],
                             "status": "access_blocked", "source_url": url, "error": blocked}
        except urllib.error.HTTPError as error:
            results[slug] = {"slug": slug, "manifest_status": manifest_row["status"]["catalog_image"],
                             "status": "page_not_found" if error.code == 404 else "fetch_or_parse_error",
                             "source_url": url, "error": f"HTTPError: HTTP {error.code}"}
        except Exception as error:
            results[slug] = {"slug": slug, "manifest_status": manifest_row["status"]["catalog_image"],
                             "status": "fetch_or_parse_error", "source_url": url,
                             "error": f"{type(error).__name__}: {error}"}
        write_jsonl(output, [results[item["wine_id"]] for item in rows if item["wine_id"] in results])
    return [results[row["wine_id"]] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--status", action="append", dest="statuses",
                        choices=["missing", "ambiguous", "candidate"],
                        help="manifest image status to audit; repeatable")
    parser.add_argument("--sample-seed", help="stable hash-order seed for a distributed sample")
    parser.add_argument("--report", type=Path, help="write candidate-audit summary JSON")
    parser.add_argument("--resume-from", type=Path, help="seed a new output with an earlier partial audit")
    parser.add_argument("--slug", action="append", dest="slugs",
                        help="audit a specific unresolved known slug; repeatable")
    args = parser.parse_args()
    patches = run(args.manifest, args.output, args.limit, args.timeout, args.inventory, args.slugs,
                  args.raw_dir, args.asset_dir, not args.no_resume, args.retry_errors,
                  set(args.statuses) if args.statuses else None, args.sample_seed, args.resume_from)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(candidate_audit_report(args.manifest, patches, args.sample_seed),
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
