# Listing-date evidence discovery — 18-Nov-2026

## Purpose and decision boundary

This note records primary-source discovery for the three symbols currently marked
`LISTING_DATE_UNKNOWN` in `data/historical_coverage_audit.csv`:
`BAGMANE`, `BIRET`, and `EMBASSY`.

It is a research lead, not an approved listing-date correction. No values in the
historical coverage audit or listing-date source data are changed by this note.
The source-document acquisition manifest remains header-only. The cited public
documents have not yet been captured as original bytes in the repository archive,
hashed, linked to reviewed registry entries, or independently reviewed under the
project's evidence protocol.

## Candidate primary-source findings

| Symbol | Candidate listing date | Candidate primary evidence | Current limitation |
|---|---|---|---|
| BAGMANE | 2026-05-14 | Bagmane Prime Office REIT half-yearly report for 30-Sep-2026, hosted in NSE corporate filings. Its listing narrative says units commenced trading on NSE and BSE on 14-May-2026. The report's stock-exchange listing table separately gives BSE scrip code 544758 and ISIN INE20VN25015. | NSE circular CML74171 / 0770/2026 dated 13-May-2026 identifies symbol BAGMANE with ISIN INE2OVN25015 and says the listing date would be announced separately. The differing ISIN strings in the two documents must be reconciled against the actual dated NSE and BSE master snapshots. Do not silently normalize or join them. |
| BIRET | 2021-02-16 | Embassy Office Parks REIT's India REIT Primer dated 13-Mar-2025, submitted to NSE/BSE and hosted in NSE archives, lists Brookfield India REIT / NSE ticker BIRET / BSE code 543261 and gives listing date 16-Feb-2021. | This is issuer-originated summary evidence, not the original listing circular or a dated security-master snapshot. Capture and cross-check the original exchange listing notice and dated masters. |
| EMBASSY | 2019-04-01 | The same issuer-originated India REIT Primer lists Embassy REIT / NSE ticker EMBASSY / BSE code 542602 and gives listing date 01-Apr-2019. A separate Embassy REIT half-yearly report hosted in NSE archives also states that its units were listed on NSE and BSE on 01-Apr-2019. | Capture the original exchange listing notice and dated masters; confirm the ISIN and venue-specific security identifiers for the historical interval. |

## Source documents for acquisition

1. BAGMANE issuer half-yearly report (NSE archive):
   https://nsearchives.nseindia.com/corporate/BAGMANE_13112026211944_BagmaneREITHalfYearlyReportSeptember302026.pdf
   Relevant locations: listing narrative in the report and the stock-exchange listing details table (PDF page 105 in the extracted report).
2. NSE circular CML74171 / 0770/2026 for BAGMANE:
   https://nsearchives.nseindia.com/content/circulars/CML74171.pdf
   Relevant location: circular dated 13-May-2026; it identifies BAGMANE and ISIN INE2OVN25015 but defers the listing date to a separate circular.
3. Issuer-originated India REIT Primer dated 13-Mar-2025, hosted in NSE archives:
   https://nsearchives.nseindia.com/corporate/EMBASSY_13032025142744_Embassy_REIT_-_Hosted_Session_for_Waterfield_Advisors.pdf
   Relevant location: page 20, "Listed REIT Landscape" table. It gives EMBASSY — 01-Apr-2019 and BIRET — 16-Feb-2021, alongside their NSE tickers and BSE scrip codes.
4. Embassy REIT half-yearly report (NSE archive), corroborating EMBASSY:
   https://nsearchives.nseindia.com/corporate/EMBASSY_02112024111645_EmbassyREITHalfYearlyReportFY25.pdf
   Relevant location: Trust Information note, which states that units were listed on NSE and BSE on 01-Apr-2019.

## Required next actions before changing listing eligibility

1. Capture each cited original document byte-for-byte; record actual retrieval URL,
   UTC retrieval time, MIME type, byte length, SHA-256, archive path, and precise
   page/table locator in `historical_source_documents.csv`.
2. Keep source registry entries at `DISCOVERY_ONLY` until a reviewer evaluates
   each document's authority scope. Do not mark a summary presentation as an
   exchange security master.
3. Obtain dated NSE and BSE security-master snapshots around each listing date.
   Resolve BAGMANE's ISIN discrepancy and confirm venue-specific identity.
4. Obtain the original exchange listing/admission circular for each venue and
   reconcile its effective date with the issuer document and master snapshots.
5. Have a second reviewer inspect the retained original bytes and approve the
   exact listing-date evidence before updating the reviewed listing-date audit.
6. Re-run listing eligibility and historical-universe reconciliation. A listing
   date does not establish historical index membership or broader universe
   eligibility.

## Current status

- Discovery: candidate dates found for all three symbols.
- Original byte capture and SHA-256: not yet done.
- Dated NSE/BSE master cross-check: not yet done.
- BAGMANE ISIN discrepancy: unresolved.
- Independent review: not done.
- Canonical listing-date audit: unchanged; all three remain `LISTING_DATE_UNKNOWN`.
- Model approval: not granted; overall research status remains `REVIEW_REQUIRED`.
