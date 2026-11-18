# Corporate-action source evidence protocol

## Purpose and current gate

This protocol governs the collection and review of dividend and stock-split events for the isolated research model. It does not certify the embedded `dividends` or `stock_splits` fields in `data/ohlcv.csv`, and it does not authorize partition generation, adjusted-price production, or model training.

The current event ledger remains header-only. Its identity columns are ISIN, historical symbol, historical series, and optional exchange security code. Do not populate it from OHLCV non-zero values, search-result snippets, commercial aggregators, inferred price jumps, or unverified vendor exports.

## Source hierarchy

Use primary evidence, retaining each source statement independently:

1. **Issuer-originated filing or shareholder notice** published by the issuer or its registrar, where it explicitly states the action terms and dates.
2. **Exchange-published disclosure or corporate-action record** for the security on NSE and/or BSE. Capture the actual record or attached document, not only a search result or rendered summary.
3. **Official exchange reference/master data** only to resolve instrument identity (for example, symbol, series, security code, and ISIN); it is not a substitute for the event announcement.
4. **Licensed exchange data feed** only where the repository has documented authorization and the original delivery artefact, schema/version, and retrieval metadata are retained.

Third-party websites may be used to discover candidate events, but never as the sole evidence for an event marked independently verified. NSE's investor-relations corporate-actions page is a corporate-actions page for NSE itself; do not mistake it for a complete market-wide historical event archive. NSE company disclosures and BSE per-security corporate-action information should be located for the specific issuer/security. BSE describes its corporate-data feed as a data product with registration and plan access; do not assume unrestricted historical API access.

## Capture procedure

For every candidate event:

1. Resolve the security to an **ISIN** using an official exchange/security master or issuer filing. Record the symbol and series as they appeared at the event date. Do not rely on today's symbol alone where a rename, merger, or ISIN change may matter.
2. Retrieve the original issuer/exchange document or machine-readable exchange record. Preserve its bytes unchanged in an evidence store outside the canonical OHLCV source.
3. Compute SHA-256 over the exact retained bytes. Record the exact source URL, retrieval timestamp in UTC, source/publisher, file type, and a stable evidence reference (document name plus page/table/row/paragraph).
4. Transcribe only fields explicitly supported by the cited evidence. Keep announcement date, ex-date, record date, and payment date distinct. Leave unsupported optional dates blank; never substitute record date for ex-date.
5. For dividends, capture amount per share and currency exactly as stated, plus any declared dividend type in review notes. Do not silently normalize units, face value, or per-share basis.
6. For splits, record the ratio as **new shares / old shares**. Retain the source's original wording and face-value terms in review notes. Do not confuse a split with a bonus issue, consolidation, rights issue, or merger.
7. If NSE and BSE or the issuer disagree, retain each source statement and mark the event unresolved. Do not overwrite one source with another or choose a value merely because it matches embedded OHLCV.
8. A second reviewer must compare the ledger row against the retained source bytes and document the review identity/time before the row can claim `INDEPENDENTLY_VERIFIED`. The validator's structural status is not verification.

## Identity, matching, and adjustment cautions

- The ledger requires a checksum-valid ISO 6166 ISIN plus the historical symbol and series; an optional security code is retained where the source provides it. Duplicate event keys use ISIN, action type, and ex-date, so a symbol rename does not silently create a second instrument identity. Symbol/series remain as event-date audit fields. This still does not model successor/predecessor relationships across mergers, demergers, or ISIN changes; those events remain out of scope and must be resolved before any cross-identifier history is joined.
- Do not match ledger events to embedded OHLCV fields until the vendor's date convention, amount/ratio encoding, adjustment state, and completeness are independently established.
- A split adjustment requires a reviewed event ratio and effective-date convention. Dividend-inclusive total returns require a separately approved cash-flow/reinvestment convention. Do not infer either from price discontinuities.
- This protocol currently covers dividends and ordinary stock splits only. Other corporate actions remain out of scope and must not be coerced into either supported type.

## Required disposition

Every candidate must end in one of: `PENDING`, `INDEPENDENTLY_VERIFIED`, or `REJECTED`. Keep the overall audit at `REVIEW_REQUIRED` until source completeness, identity resolution, conflicts, and embedded-field semantics have been reviewed. A structurally valid ledger is never sufficient to clear this gate.

## Public source references

- NSE investor-relations corporate-actions page (NSE's own corporate actions): https://www.nse.in/static/investor-relations/corporate-actions
- NSE corporate-action adjustment overview (useful for understanding action categories and adjustment concepts; not a substitute for issuer event evidence): https://www.nseindia.com/products-services/equity-derivatives-corporate-actions-adjustments
- BSE shareholder FAQ identifying the BSE corporate-action information page: https://www.bseindia.com/downloads1/FAQ_of_shareholders.pdf
- BSE market-data portal describing its corporate-data product and access process: https://marketdata.bseindia.com/

These references establish discovery routes and context only. They do not supply event records for this repository or verify any row in its ledger.


## Point-in-time security identity history

The empty template `research_model/schemas/security_identity_history.csv` records the identity that an official source associates with an instrument over a stated validity interval: ISIN, historical symbol, series, optional security code, inclusive `valid_from`/`valid_to` dates, and source-document provenance. An empty template is intentional; no historical mappings are assumed.

Run its structural check:

```bash
python -m research_model.data.validate_identity_history research_model/schemas/security_identity_history.csv
```

The validator checks ISIN checksums, date order, required identity and evidence fields, UTC retrieval/review metadata, and overlapping validity intervals for the same venue/ISIN. It also rejects concurrent claims where the same venue/symbol/series or non-empty venue/security code maps to different ISINs; sequential reuse after an interval ends is allowed. It always reports `REVIEW_REQUIRED`; it cannot authenticate the source document or verify that the stated interval is true.

A symbol/series change may be represented as consecutive intervals only when an authoritative source establishes the effective dates. Different ISINs are always treated as separate identities. Mergers, demergers, amalgamations, and other successor/predecessor relationships are not inferred or joined by this tool. They require a separately reviewed corporate-reorganization record and explicit methodology before any historical price series or membership history is linked.
