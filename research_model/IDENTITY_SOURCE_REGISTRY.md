# Identity source registry

## Step 15 — authoritative identity-source registry

This capability inventories candidate primary sources from which identity evidence
may later be collected. A registry row is a discovery and governance record, not
proof of any security's ISIN, symbol, series, security code, or effective dates.

### Current discovery inventory

The registry contains three candidate routes, all deliberately marked
`DISCOVERY_ONLY`:

- **NSE historical Masters documentation** — NSE's historical data dissemination
  document describes a Masters database containing symbol, series, and ISIN
  reference data. The actual dated master files, schemas, archive paths, and
  historical coverage still need to be acquired and checked.
- **NSE daily MII security-file catalogue** — the current All Reports catalogue
  exposes dated NSE listed-security files, including a combined NSE-listed and
  BSE-exclusive option. This is a current-snapshot discovery route, not proof
  that historical snapshots are available or complete.
- **BSE standardized security-master specification** — the specification defines
  fields including scrip code, instrument code, group, scrip name, and ISIN.
  It documents the intended schema; no dated historical BSE master snapshots
  have yet been acquired or verified.

These entries are source-discovery leads only. They do not establish any
security-level mapping, historical effective interval, or complete point-in-time
coverage. Do not change their status to `REVIEWED_AUTHORITATIVE` merely because
the landing page or format specification is published by an exchange.

### Source classes

- `EXCHANGE_SECURITY_MASTER`: exchange-published security master snapshots.
- `EXCHANGE_CIRCULAR`: exchange circulars defining symbol, series, or security events.
- `LISTING_NOTICE`: official exchange listing/admission notices.
- `ISSUER_REGULATORY_FILING`: issuer disclosures filed through a regulated channel.
- `OTHER_PRIMARY_SOURCE`: exceptional primary sources requiring explicit review.

The venue must be `NSE`, `BSE`, `ISSUER`, or `OTHER`. Candidate URLs must be
HTTPS. `DISCOVERY_ONLY` is the default status. Marking a source
`REVIEWED_AUTHORITATIVE` requires a named reviewer and UTC review timestamp;
that decision evaluates the source's stated authority scope only.

### Evidence separation

1. Preserve the original downloaded document separately from this registry.
2. Record its exact URL, retrieval timestamp, and SHA-256 in the evidence record.
3. Record the document's own publication/effective date separately from retrieval time.
4. Cite the exact page, row, circular number, or filing reference supporting each fact.
5. Require independent human review before accepting a security identity interval.
6. Never infer continuity across different ISINs. Mergers, demergers, and successor/
   predecessor links require a separate corporate-action review.

A registry validator result always remains `REVIEW_REQUIRED`; it cannot authenticate
a document, establish historical coverage, approve an identity mapping, or authorize
historical partition generation. The registry is no longer empty, but every current
entry remains discovery-only until the underlying dated files and source-specific
retrieval procedures are reviewed.

Run:

```bash
python -m research_model.data.validate_identity_sources research_model/schemas/identity_source_registry.csv
pytest -q tests/research_model/test_validate_identity_sources.py
```
