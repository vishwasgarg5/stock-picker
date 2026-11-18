"""Fetch allowlisted primary-source PDFs and append immutable capture metadata.

Downloads are byte-preserved. Capture never certifies source authority or review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "research_model/schemas/identity_source_registry.csv"
MANIFEST = ROOT / "research_model/schemas/historical_source_documents.csv"
ARCHIVE = ROOT / "research_data/source_documents"
MAX_BYTES = 25 *  1024 * 1024
ALLOWED_HOSTS = {"nsearchives.nseindia.com", "www.nseindia.com", "www.bseindia.com", "www.bseindia.in"}
# Exact URL allowlist: no URLs supplied by workflow inputs or user-controlled data.
SOURCES = (
    ("NSE_DAILY_MII_FILE_DISSEMINATION_CIRCULAR", "NSE_MSD60315_20240119", "MSD60315.pdf",
     "https://nsearchives.nseindia.com/content/circulars/MSD60315.pdf",
     "NSE circular dated 2024-01-19; dissemination effective 2024-02-05"),
    ("NSE_MASTER_DATA_SPECIFICATION", "NSE_MASTERS_SPECIFICATION", "NSE-Masters-Data-v1.6.pdf",
     "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/NSE-Masters%20Data-v1.6.pdf",
     "Version/date and field semantics require inspection of original PDF"),
    ("BSE_STANDARD_SECURITY_MASTER_FORMAT", "BSE_SECURITY_MASTER_FORMAT", "BSE-standard-security-master-format.pdf",
     "https://www.bseindia.com/markets/MarketInfo/DownloadAttach.aspx?attachedId=e99e2ca5-95d7-4173-83da-7774412bbb2d&id=20220531-44",
     "Specification document date and scope require inspection of original PDF"),
    ("NSE_HISTORICAL_MASTERS_DOCUMENTATION", "NSE_HISTORICAL_CM_DATA_NOTE", "NSE-historical-CM-data-note.pdf",
     "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/Data_details_CM_1.pdf",
     "Historical archive structure note; does not itself supply a dated master snapshot"),
)

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise urllib.error.HTTPError(newurl, code, "Redirect outside HTTPS source allowlist", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def _download(url: str) -> tuple[bytes, str, str]:
    opener = urllib.request.build_opener(SafeRedirect())
    request = urllib.request.Request(url, headers={
        "User-Agent": "stock-picker-research-evidence-capture/1.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    })
    with opener.open(request, timeout=45) as response:
        final_url = response.geturl()
        parsed = urlparse(final_url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise RuntimeError(f"Final URL outside HTTPS source allowlist: {final_url}")
        payload = response.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise RuntimeError(f"Document exceeds {MAX_BYTES} byte safety limit: {url}")
        content_type = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].strip().lower()
        last_modified = response.headers.get("Last-Modified", "")
    if not payload.startswith(b"%PDF-"):
        raise RuntimeError(f"Response is not a PDF (possible HTML/error page): {url}")
    return payload, final_url, content_type or "application/pdf"

def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)

def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def capture() -> dict:
    registry_fields, registry_rows = _read_csv(REGISTRY)
    manifest_fields, manifest_rows = _read_csv(MANIFEST)
    required = {"source_id", "venue", "source_class", "canonical_url", "review_status"}
    if not required.issubset(registry_fields):
        raise RuntimeError(f"Source registry missing columns: {sorted(required - set(registry_fields))}")
    by_id = {row["source_id"]: row for row in registry_rows}
    if len(by_id) != len(registry_rows):
        raise RuntimeError("Duplicate source IDs in registry")
    existing_ids = {row.get("document_id", "") for row in manifest_rows}
    existing_paths = {row.get("archive_path", "") for row in manifest_rows}
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for source_id, doc_id, filename, url, note in SOURCES:
        try:
            source = by_id.get(source_id)
            if not source:
                raise RuntimeError(f"Source ID absent from registry: {source_id}")
            if source["canonical_url"].strip() != url:
                raise RuntimeError(f"Allowlisted URL differs from registry for {source_id}")
            if source["review_status"].strip() != "DISCOVERY_ONLY":
                raise RuntimeError(f"Refusing to alter non-discovery source status: {source_id}")
            if doc_id in existing_ids:
                results.append({"source_id": source_id, "status": "ALREADY_CAPTURED"})
                continue
            payload, final_url, content_type = _download(url)
            digest = hashlib.sha256(payload).hexdigest()
            relative = f"{source_id}/{digest[:16]}-{filename}"
            if relative in existing_paths:
                raise RuntimeError(f"Archive path already exists without matching document ID: {relative}")
            target = ARCHIVE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never overwrite a previously captured object.
            with target.open("xb") as handle:
                handle.write(payload)
            row = {
                "document_id": doc_id, "source_id": source_id,
                "venue": source["venue"], "source_class": source["source_class"],
                "document_title": source["source_name"],
                "canonical_url": url, "retrieved_url": final_url,
                "publication_date": "", "effective_from": "", "effective_to": "",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "content_type": content_type, "byte_length": str(len(payload)),
                "sha256": digest, "archive_path": relative,
                "evidence_locator": "Whole original PDF captured; page-level evidence locator pending review",
                "capture_method": "GitHub Actions HTTPS GET; original response bytes preserved",
                "capture_status": "CAPTURED_UNREVIEWED", "reviewed_by": "",
                "reviewed_at_utc": "", "notes": note,
            }
            manifest_rows.append(row)
            existing_ids.add(doc_id)
            existing_paths.add(relative)
            results.append({"source_id": source_id, "status": "CAPTURED",
                            "bytes": len(payload), "sha256": digest, "archive_path": relative})
        except Exception as exc:
            failures.append({"source_id": source_id, "error": str(exc)})
    if results and any(item["status"] == "CAPTURED" for item in results):
        _write_csv(MANIFEST, manifest_fields, manifest_rows)
    report = {"captured": sum(x["status"] == "CAPTURED" for x in results),
              "already_captured": sum(x["status"] == "ALREADY_CAPTURED" for x in results),
              "failed": len(failures), "results": results, "failures": failures,
              "source_review_promoted": False, "independent_review_claimed": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate registry/manifest without network access")
    args = parser.parse_args()
    if args.check_only:
        _, registry = _read_csv(REGISTRY)
        _, manifest = _read_csv(MANIFEST)
        ids = [r["source_id"] for r in registry]
        if len(ids) != len(set(ids)):
            raise SystemExit("Duplicate source IDs")
        if any(r.get("capture_status") == "INDEPENDENTLY_REVIEWED" and
               (r.get("reviewed_by") or r.get("reviewed_at_utc")) for r in manifest):
            raise SystemExit("Malformed review metadata")
        print(f"Registry rows: {len(registry)}; captured document rows: {len(manifest)}")
        return
    capture()

if __name__ == "__main__":
    main()
