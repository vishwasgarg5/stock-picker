"""Discover, but never guess, official NSE dated MII security-file download routes.

The NSE catalogue is JavaScript-backed. This probe records only links and script
references actually published by the official catalogue; it does not synthesize
a dated file URL or promote a source.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

CATALOGUE = "https://www.nseindia.com/all-reports"
ALLOWED_HOSTS = {"www.nseindia.com", "nseindia.com", "www.nseindia.in", "nseindia.in",
                 "nsearchives.nseindia.com"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])


def get(url: str) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise RuntimeError(f"Refusing non-allowlisted official URL: {url}")
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 stock-picker-research-evidence-discovery/1.0",
        "Accept": "text/html,application/javascript,*/*;q=0.8",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        final = response.geturl()
        final_parsed = urlparse(final)
        if final_parsed.scheme != "https" or final_parsed.hostname not in ALLOWED_HOSTS:
            raise RuntimeError(f"Redirect outside official NSE host allowlist: {final}")
        return response.read(8 * 1024 * 1024 + 1), final


def main():
    payload, final_url = get(CATALOGUE)
    if len(payload) > 8 * 1024 * 1024:
        raise RuntimeError("NSE catalogue exceeds 8 MiB probe limit")
    html = payload.decode("utf-8", errors="replace")
    parser = Links()
    parser.feed(html)
    hrefs = [urljoin(final_url, href) for href in parser.links]
    relevant_links = [u for u in hrefs if re.search(r"(mii|security|scrip|master)", u, re.I)]
    script_urls = [urljoin(final_url, src) for src in parser.scripts]
    relevant_scripts = []
    for url in script_urls:
        host = urlparse(url).hostname
        if host not in ALLOWED_HOSTS:
            continue
        try:
            body, actual = get(url)
        except (OSError, urllib.error.URLError, RuntimeError):
            continue
        source = body.decode("utf-8", errors="replace")
        hits = []
        for match in re.finditer(r".{0,180}(?:mii|security|scrip|all-reports|reportMaster|download).{0,220}", source, re.I):
            hits.append(match.group(0)[:420])
            if len(hits) >= 30:
                break
        if hits:
            relevant_scripts.append({"url": actual, "matches": hits})
    report = {
        "catalogue_url": final_url,
        "catalogue_bytes": len(payload),
        "catalogue_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "catalogue_relevant_links": sorted(set(relevant_links)),
        "official_script_count": len(script_urls),
        "script_matches": relevant_scripts,
        "report_variants_seen_in_catalogue_text": {
            "nse_listed": "CM - MII - Security File (.gz) (NSE Listed securities)" in html,
            "nse_and_bse_exclusive": "CM - MII - Security File (.gz) (NSE Listed and BSE Exclusive securities)" in html,
        },
        "download_url_synthesized": False,
        "source_status_promoted": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not any(report["report_variants_seen_in_catalogue_text"].values()):
        raise SystemExit("Expected MII report labels not found; catalogue shape may have changed")


if __name__ == "__main__":
    main()
