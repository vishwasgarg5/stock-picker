"""Discover official NSE MII security-file routes without guessing URLs.

The probe reads the official All Reports page, its inline JavaScript, and its
allowlisted script assets. It reports endpoint strings actually present in
those responses; it never synthesizes a dated download URL or downloads a
security-master file.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

CATALOGUE = "https://www.nseindia.com/all-reports"
ALLOWED_HOSTS = {
    "www.nseindia.com", "nseindia.com", "www.nseindia.in",
    "nseindia.in", "nsearchives.nseindia.com",
}
MAX_PAGE = 8 * 1024 * 1024
MAX_SCRIPT = 5 * 1024 * 1024
MAX_SCRIPTS = 60
TERMS = re.compile(
    r"(?:mii|eq_security|security.?file|scrip.?master|report.?detail|"
    r"report.?master|download.?report|api/|/api|\.csv|\.gz)",
    re.I,
)
ENDPOINT = re.compile(
    r"""(?P<q>["'`])(?P<path>(?:https?://[^"'\s`]+|/[^"'\s`]+|(?:api|reports?)[^"'\s`]*))(?P=q)""",
    re.I,
)


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.inline_scripts: list[str] = []
        self._in_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "script":
            if attrs.get("src"):
                self.scripts.append(attrs["src"])
            else:
                self._in_script = True
                self._script_parts = []

    def handle_data(self, data):
        if self._in_script:
            self._script_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._in_script:
            self.inline_scripts.append("".join(self._script_parts))
            self._in_script = False
            self._script_parts = []


def get(url: str, limit: int) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise RuntimeError(f"Refusing non-allowlisted official URL: {url}")
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 stock-picker-research-evidence-discovery/1.1",
        "Accept": "text/html,application/javascript,*/*;q=0.8",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        final = response.geturl()
        final_parsed = urlparse(final)
        if final_parsed.scheme != "https" or final_parsed.hostname not in ALLOWED_HOSTS:
            raise RuntimeError(f"Redirect outside official NSE host allowlist: {final}")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise RuntimeError(f"Response exceeds {limit} byte probe limit: {final}")
        return body, final


def snippets(source: str, pattern: re.Pattern[str], limit: int = 80) -> list[str]:
    found = []
    seen = set()
    for match in pattern.finditer(source):
        item = source[max(0, match.start() - 220):min(len(source), match.end() + 320)]
        item = re.sub(r"\s+", " ", item).strip()
        if item not in seen:
            seen.add(item)
            found.append(item[:650])
        if len(found) >= limit:
            break
    return found


def main():
    payload, final_url = get(CATALOGUE, MAX_PAGE)
    html = payload.decode("utf-8", errors="replace")
    parser = PageParser()
    parser.feed(html)
    hrefs = [urljoin(final_url, href) for href in parser.links]
    relevant_links = sorted({
        url for url in hrefs
        if urlparse(url).hostname in ALLOWED_HOSTS and TERMS.search(url)
    })

    assets = []
    for src in parser.scripts[:MAX_SCRIPTS]:
        url = urljoin(final_url, src)
        if urlparse(url).hostname in ALLOWED_HOSTS and url not in assets:
            assets.append(url)

    inspected = []
    for inline_index, source in enumerate(parser.inline_scripts):
        if TERMS.search(source):
            inspected.append({
                "kind": "inline",
                "url": f"{final_url}#inline-script-{inline_index}",
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "endpoint_like_strings": snippets(source, ENDPOINT, 30),
                "relevant_context": snippets(source, TERMS, 35),
            })

    script_errors = []
    for url in assets:
        try:
            body, actual = get(url, MAX_SCRIPT)
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            script_errors.append({"url": url, "error": str(exc)[:240]})
            continue
        source = body.decode("utf-8", errors="replace")
        if TERMS.search(source):
            inspected.append({
                "kind": "external",
                "url": actual,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "endpoint_like_strings": snippets(source, ENDPOINT, 40),
                "relevant_context": snippets(source, TERMS, 45),
            })

    report = {
        "catalogue_url": final_url,
        "catalogue_bytes": len(payload),
        "catalogue_sha256": hashlib.sha256(payload).hexdigest(),
        "catalogue_relevant_links": relevant_links,
        "official_script_count": len(parser.scripts),
        "inline_script_count": len(parser.inline_scripts),
        "scripts_inspected": len(assets),
        "script_errors": script_errors[:20],
        "route_evidence": inspected,
        "report_variants_seen_in_catalogue_text": {
            "nse_listed": "CM - MII - Security File (.gz) (NSE Listed securities)" in html,
            "nse_and_bse_exclusive": "CM - MII - Security File (.gz) (NSE Listed and BSE Exclusive securities)" in html,
        },
        "download_url_synthesized": False,
        "security_file_downloaded": False,
        "source_status_promoted": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not any(report["report_variants_seen_in_catalogue_text"].values()):
        raise SystemExit("Expected MII report labels not found; catalogue shape may have changed")


if __name__ == "__main__":
    main()
