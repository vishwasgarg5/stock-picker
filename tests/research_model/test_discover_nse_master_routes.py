from research_model.data.discover_nse_master_routes import (
    ALLOWED_HOSTS,
    ENDPOINT,
    PageParser,
    get,
    snippets,
)


def test_page_parser_collects_report_links_and_inline_scripts():
    parser = PageParser()
    parser.feed(
        '<a href="/report-detail/eq_security">MII Security File</a>'
        '<script src="/assets/report.js"></script>'
        '<script>const route = "/api/report/download";</script>'
    )

    assert parser.links == ["/report-detail/eq_security"]
    assert parser.scripts == ["/assets/report.js"]
    assert parser.inline_scripts == ['const route = "/api/report/download";']


def test_endpoint_discovery_extracts_only_literal_route_strings():
    source = (
        'const route = "/api/report/download"; '
        'const detail = "https://www.nseindia.com/report-detail/eq_security";'
    )

    paths = [match.group("path") for match in ENDPOINT.finditer(source)]

    assert paths == [
        "/api/report/download",
        "https://www.nseindia.com/report-detail/eq_security",
    ]


def test_snippets_collapse_whitespace_and_deduplicate():
    result = snippets("prefix\n  MII security file\n suffix", __import__("re").compile("MII", __import__("re").I))

    assert len(result) == 1
    assert "\n" not in result[0]
    assert "MII security file" in result[0]


def test_get_rejects_non_https_or_non_allowlisted_hosts_before_network():
    import pytest

    with pytest.raises(RuntimeError, match="Refusing non-allowlisted official URL"):
        get("https://example.com/secret", 1024)
    with pytest.raises(RuntimeError, match="Refusing non-allowlisted official URL"):
        get("http://www.nseindia.com/all-reports", 1024)

    assert "www.nseindia.com" in ALLOWED_HOSTS
