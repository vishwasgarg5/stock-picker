# Historical security-master document acquisition

## Purpose and boundary

The acquisition manifest at `schemas/historical_source_documents.csv` records exact source-document provenance for later point-in-time identity review. It is deliberately separate from the identity-source registry: a registry entry describes a candidate source; a document row describes one captured artifact. Neither approves a security mapping.

No live scraping, scheduled download, guessed archive URL, or automatic source certification is part of this capability. Capture only documents actually obtained from a source you are authorized to access. The committed manifest remains header-only until genuine documents are captured.

## Capture procedure

For each document:

1. Resolve its `source_id` to a reviewed source-registry entry. Preserve the venue, source class, exact document title, canonical discovery URL, and actual retrieval URL.
2. Save the original response bytes without normalizing line endings, re-encoding text, opening and re-saving a spreadsheet, or editing the document. Keep one immutable copy per retrieval.
3. Record publication date and the document's effective-from/effective-to dates separately. Record the retrieval timestamp in UTC; retrieval time is not an effective date.
4. Record the MIME/content type, exact byte length, SHA-256 of the original bytes, safe relative archive path, capture method, and a precise evidence locator (page, table, row, circular number, or filing section).
5. Keep the archived document and manifest together under a controlled archive root. Do not overwrite an earlier capture; a changed document is a new artifact with a new identifier and hash.
6. A second reviewer checks the source's authority scope, document completeness, date semantics, and the cited fields against the original bytes. Record reviewer identity and UTC review time only after that review.

Suggested archive convention (not a populated dataset):

```text
research_data/
  source_documents/
    nse/
      security_masters/
        YYYY/
          <document-id>.<original-extension>
  manifests/
    historical_source_documents.csv
```

The original extension may be retained as part of the archive filename. The manifest's `archive_path` is relative to the archive root passed to the validator.

## Validation

Metadata-only structural check:

```bash
python -m research_model.data.validate_source_archive research_model/schemas/historical_source_documents.csv --source-registry research_model/schemas/identity_source_registry.csv
```

Verify actual archived bytes against manifest size and SHA-256:

```bash
python -m research_model.data.validate_source_archive research_model/schemas/historical_source_documents.csv --source-registry research_model/schemas/identity_source_registry.csv --archive-root /path/to/research_data/source_documents
```

When supplied, the source registry is used to reject document rows whose source ID is not registered, whose venue/source class/canonical URL disagree with the registry, or whose independently reviewed capture is attached to a source that is not itself marked `REVIEWED_AUTHORITATIVE` with reviewer metadata. The validator rejects duplicate document IDs, malformed HTTPS URLs, invalid dates/UTC timestamps, reversed effective intervals, unsafe paths, missing files, byte-length mismatches, and SHA-256 mismatches. It does not fetch URLs or modify source documents. CI uses only the empty template and deterministic synthetic fixtures; it must not download exchange data.

## Evidence and approval limits

- A matching SHA-256 proves byte integrity relative to the manifest, not that the source is genuine, complete, or authoritative.
- A reviewed source registry entry does not prove that a particular archived document supports a particular identity interval. The validator requires an authoritative reviewed source entry before a document can be marked independently reviewed, but that status still requires a human to inspect the source scope.
- A reviewed document does not itself validate an ISIN/symbol/series mapping. Every mapping needs a precise document locator, source-specific date semantics, and independent human review.
- Do not infer effective dates from file names, retrieval dates, or the first/last observation in a dataset.
- Preserve conflicting documents and record the conflict; do not silently choose one.
- Do not merge different ISINs or infer successor/predecessor relationships from symbol similarity.
- No identity mapping, corporate-action event, point-in-time universe membership, or historical partition is authorized by this acquisition contract.

The point-in-time identity ledger is venue-scoped: every row must carry both `venue` (`NSE` or `BSE`) and the exact `source_id`. The same ISIN may therefore have separate NSE and BSE identity intervals without being treated as an overlapping duplicate. When a document manifest is supplied, the validator also requires the identity row's venue and source ID to match the referenced document. This keeps native exchange identity separate from cross-venue interoperability evidence.

For a stricter identity-evidence check, pass the captured-document manifest to the identity validator:

```bash
python -m research_model.data.validate_identity_history research_model/schemas/security_identity_history.csv --source-documents research_model/schemas/historical_source_documents.csv
```

With this option, each identity row's document SHA-256 must exist in the manifest,
the referenced document must be marked `INDEPENDENTLY_REVIEWED`, and the identity
row's source URL must match that manifest entry's canonical or retrieval URL. This
is a provenance-link check only: the validator still does not inspect the archived
bytes, establish the truth of a mapping, or replace independent review of the
identity interval. Run the source-archive validator with `--archive-root` separately
to verify archived bytes.

The capability always reports `REVIEW_REQUIRED`; archive-integrity success is not a source-authority or research-readiness approval.


## Exchange archive access findings — 18-Nov-2026

The source-discovery check found an important access distinction. Do not assume that a public report catalogue is a public historical security-master archive.

### NSE

- The [NSE All Reports catalogue](https://www.nseindia.com/all-reports) currently lists a daily CM MII security file named `NSE_CM_security_DDMMYYYY.csv.gz`. This establishes a current daily discovery route; it does not establish that every historical date can be retrieved from the same catalogue.
- NSE circular [MSD60315 (19-Jan-2024)](https://nsearchives.nseindia.com/content/circulars/MSD60315.pdf) says the CM MII security master file is disseminated daily on the website from 05-Feb-2024, with the above filename pattern. Retain the circular as dissemination-process evidence, not as a security snapshot.
- NSE Data & Analytics' [Masters Data specification](https://nsearchives.nseindia.com/web/sites/default/files/inline-files/NSE_MasterData_Technical_Specifications.pdf) describes an EOD security/contract-master product distributed to vendors through SFTP using agreement-period credentials. This is a distinct access channel; do not assume it is anonymously downloadable.
- NSE's [historical capital-market data dissemination note](https://nsearchives.nseindia.com/content/press/Data_details_CM.pdf) describes a historical-data structure with a Masters directory containing symbols, series and ISINs. It does not by itself provide a verified public URL, entitlement, date range, or downloadable master snapshot.

**Acquisition decision:** the daily website route is a candidate for snapshots from its documented dissemination start date onward. Historical pre-05-Feb-2024 masters require confirmation of the historical-data product's access terms and actual archive availability. No archive URL or file is inferred here.

### BSE

- BSE's [standardized security-master format document](https://www.bseindia.com/markets/MarketInfo/DownloadAttach.aspx?attachedId=e99e2ca5-95d7-4173-83da-7774412bbb2d&id=20220531-44) specifies the filename pattern `BSE_EQ_SCRIP_DDMMYYYY.csv` and fields including scrip code, instrument code, group, scrip name and ISIN.
- The format specification is not a dated snapshot and does not establish an accessible historical archive route or coverage period.

**Acquisition decision:** locate and verify the official dated BSE files and their access conditions before registering a retrieval URL or capturing data. Do not construct a URL from the filename pattern.

### Required evidence before the next identity milestone

1. Confirm access rights and the official retrieval route separately for each venue and date range.
2. Obtain at least one actual dated original file per source route; preserve its exact bytes, response metadata, retrieval timestamp and SHA-256.
3. Inspect the original header/record layout and establish what date the file represents (publication date versus effective date).
4. Compare a sample against an independent official circular/listing notice before source review.
5. Only then move the corresponding source-registry entry from `DISCOVERY_ONLY` to review; never upgrade a source based only on a landing page or specification.

At this checkpoint the document manifest remains intentionally header-only. No dated NSE/BSE master snapshot has been acquired by this discovery check, and no security identity row or historical coverage claim is authorized.


### Additional NSE clarification — combined NSE/BSE-exclusive file

NSE circular **MSD67344 dated 28-Mar-2025** (Circular 13/2025) clarifies that the All Reports catalogue exposes two distinct MII security-file variants:

- NSE-listed securities; and
- NSE-listed plus BSE-exclusive securities.

The circular states that the BSE-exclusive records are provisioned at NSE for the alternative-trading-venue arrangement, effective 01-Apr-2025. It describes a BSE-exclusive symbol suffix ($) and a dedicated permitted-to-trade indicator. This is **not** evidence that the combined file is a complete BSE master or a substitute for BSE's own dated scrip master. Its purpose and population are specific to the interoperability arrangement.

The source registry now assigns separate source IDs to these populations: `NSE_DAILY_MII_SECURITY_FILE` is reserved for the NSE-listed-only variant, while `NSE_DAILY_MII_BSE_INTEROPERABILITY_FILE` is reserved for the combined NSE-listed plus BSE-exclusive variant. When capturing a dated file, select the source ID matching the exact report variant; do not use the generic catalogue page as evidence that the populations are interchangeable.

Consequences for this project:

- Treat the combined file as a separate NSE-published interoperability snapshot, not as the BSE venue's authoritative master.
- Do not assume that its BSE-exclusive population covers every BSE-listed security or any date before its effective date.
- Keep NSE-native identity, BSE-native identity, and cross-venue interoperability eligibility as distinct facts.
- Never strip the $ suffix or merge a BSE symbol with an NSE symbol without an explicit, dated source mapping.
- Retain the circular as process/field-semantics evidence; a downloaded dated file is still required for any snapshot-level claim.

Source: [NSE circular MSD67344, 28-Mar-2025](https://nsearchives.nseindia.com/content/circulars/MSD67344.pdf), especially sections A, C and D.


### Listing-date evidence leads found — 18-Nov-2026

A focused primary-document search found issuer/exchange-hosted records for all three symbols currently marked `LISTING_DATE_UNKNOWN` in `data/historical_coverage_audit.csv`. These are **candidate evidence leads**, not yet accepted identity-ledger rows: the files have not been captured into the immutable source archive, their bytes have not been hashed in the manifest, and no independent reviewer has approved the mapping.

| Symbol | Candidate listing date | Primary document lead | What it supports | Current treatment |
|---|---|---|---|---|
| `BAGMANE` | 2026-05-14 | [NSE listing circular CML74171, 13-May-2026](https://nsearchives.nseindia.com/content/circulars/CML74171.pdf); [Bagmane Prime Office REIT half-yearly report](https://nsearchives.nseindia.com/corporate/BAGMANE_13112026211944_BagmaneREITHalfYearlyReportSeptember302026.pdf) | NSE circular identifies symbol BAGMANE and ISIN INE2OVN25015; issuer report states units commenced trading on NSE and BSE on 14-May-2026 | `DISCOVERY_ONLY`; confirm exact NSE effective date in circular annexure and capture original documents |
| `BIRET` | 2021-02-16 | [Brookfield India REIT FY2020-21 annual report filed with NSE](https://nsearchives.nseindia.com/corporate/BIRET_26072021174205_SubmissionofIstAnnualReportandIstNoticeofUnitholdersmeeting.pdf); [NSE listing circular CML47340, 15-Feb-2021 (copy located outside NSE archive)](https://www.steelcitynettrade.com/Circulars/Listing%20of%20Units%20of%20Brookfield.pdf) | Issuer annual report explicitly states units were listed on both NSE and BSE on 16-Feb-2021; circular copy states NSE admission effective 16-Feb-2021 | `DISCOVERY_ONLY`; capture official issuer filing and locate the original NSE-hosted circular before independent approval |
| `EMBASSY` | 2019-04-01 | [Embassy Office Parks REIT FY2019-20 half-yearly report](https://www.embassyofficeparks.com/wp-content/uploads/2026/08/half-yearly-report_fy2019-20-1.pdf) | Issuer report states the REIT listed on NSE and BSE on 01-Apr-2019 | `DISCOVERY_ONLY`; capture original report bytes and independently review |

These records materially narrow the unresolved listing-date research. They do **not** authorize changing `listing_history_status` from `LISTING_DATE_UNKNOWN`, populating `listing_date`, or creating point-in-time security identity intervals. The next gate is document capture plus exact source/document review. In particular, the 2024 Brookfield NSE circular CML62583 is a later additional-issue listing event and must not be mistaken for the original 2021 IPO listing date.


### Follow-up primary-document cross-check — 18-Nov-2026

A second discovery pass distinguished the initial announcement circulars from the actual effective-date notices. These are still discovery leads only: no response bytes have been captured into the immutable archive, no SHA-256 values have been recorded, and no reviewer has approved an identity interval.

| Symbol | Better primary evidence lead | Supported fact | Remaining limitation |
|---|---|---|---|
| BAGMANE | NSE circular CML74204 (13-May-2026), https://nsearchives.nseindia.com/content/circulars/CML74204.pdf | Trading commences with effect from 14-May-2026; use the annexure to inspect the symbol, ISIN and exchange security code. | CML74171 is only the forthcoming-listing announcement and explicitly says the listing date will be notified separately. Capture and review CML74204, not CML74171 alone, for the effective date. |
| BIRET | NSE-filed Brookfield India REIT annual report, https://nsearchives.nseindia.com/corporate/BIRET_26072021174205_SubmissionofIstAnnualReportandIstNoticeofUnitholdersmeeting.pdf | The report states the REIT was listed on BSE and NSE on 16-Feb-2021. | Issuer filing is useful corroboration, but obtain the original NSE admission/listing circular and inspect its annexure for symbol, series, ISIN and effective date before mapping. |
| EMBASSY | BSE-hosted FY2019-20 annual report, https://www.bseindia.com/bseplus/AnnualReport/542602/5426020319.pdf; NSE-filed financial statement, https://nsearchives.nseindia.com/corporate/listcontract1_1162019134328_IntimationofFinancials31032019CompleteSet_231.pdf | The BSE-hosted report states listing on NSE and BSE on 01-Apr-2019; the NSE filing independently states listing on 01-Apr-2019. | These issuer reports corroborate the date but do not replace venue-specific admission notices or a dated security-master snapshot for symbol/series/ISIN intervals. |

Discovery cross-check references:
- NSE CML74171: https://nsearchives.nseindia.com/content/circulars/CML74171.pdf (announcement only; date deferred to separate circular).
- NSE CML74204: https://nsearchives.nseindia.com/content/circulars/CML74204.pdf (effective-date notice).
- NSE-filed Brookfield annual report: https://nsearchives.nseindia.com/corporate/BIRET_26072021174205_SubmissionofIstAnnualReportandIstNoticeofUnitholdersmeeting.pdf.
- BSE-hosted Embassy annual report: https://www.bseindia.com/bseplus/AnnualReport/542602/5426020319.pdf.
- NSE-filed Embassy financial statement: https://nsearchives.nseindia.com/corporate/listcontract1_1162019134328_IntimationofFinancials31032019CompleteSet_231.pdf.

No listing-date audit CSV, source registry status, document manifest, or identity-history row is changed by this discovery note. Treat all three dates as unapproved until the original documents are captured, byte-verified, and independently reviewed. A listing date alone does not establish continuous identity or index membership.
