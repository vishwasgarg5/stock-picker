# Six exchange-specific pre-listing anomaly investigations

**Investigation date:** 2026-11-18  
**PR:** #41 — research/clean-model-foundation  
**Scope:** CAMS, DALBHARAT, FORCEMOT, NESTLEIND, PATANJALI, TECHNOE

## Executive finding

These are not six ordinary incorrect listing dates. The audit compares one NSE listing-date field with OHLCV history that has no row-level exchange/provider provenance. That conflates (1) first listing on any venue, (2) venue-specific formal listing, (3) permitted-to-trade status, (4) predecessor-security history, and (5) successor securities issued after corporate reorganisation.

The six cases have credible official explanations for why some observations precede the current NSE listing date. None of those explanations, by itself, proves the venue/provider origin or correct corporate-action adjustment of each OHLCV row. This investigation resolves event interpretation, not row-level price provenance.

## Audit baseline

| Symbol | First retained OHLCV date | Current NSE listing-master date | Pre-listing interpretation |
|---|---:|---:|---|
| CAMS | 2020-10-05 | 2021-05-07 | BSE listing preceded NSE formal listing; NSE permitted-to-trade existed before formal NSE listing |
| DALBHARAT | 2018-12-21 | 2019-01-22 | Listing/trading permission for shares under restructuring began 2018-12-21; a second allotment block began 2019-01-25 |
| FORCEMOT | 2019-08-19 | 2024-02-14 | BSE history predates NSE admission; NSE date is not first-ever exchange eligibility |
| NESTLEIND | 2018-01-01 | 2023-08-01 | BSE history and NSE permitted-to-trade preceded formal NSE listing |
| PATANJALI | 2018-01-01 | 2020-01-24 | Ruchi Soya history is separated from 2020 recommencement after capital reduction/consolidation; later symbol/name change |
| TECHNOE | 2018-01-01 | 2018-12-04 | Predecessor security TECHNOELEC was listed before suspension and successor issue/listing under TECHNOE |

The OHLCV schema currently has no exchange, provider, vendor ticker, or source-document key. The existing read-only provenance audit explicitly detects this absence. Public corporate/exchange events cannot identify the source venue of an individual Yahoo-normalised bar.

## Case findings and dispositions

### 1. CAMS — BSE first, NSE permitted-to-trade, then NSE formal listing

**Official facts**
- CAMS states that its shares were listed on BSE effective 2020-10-01.
- CAMS informed BSE on 2021-05-05 that NSE formal listing/admission would take effect 2021-05-07.
- The same letter explicitly says the shares had already been trading in NSE's Permitted Category.
- The checked OHLCV history begins 2020-10-05, after BSE listing but before NSE formal listing.

**Interpretation:** 2021-05-07 is an NSE formal-listing date, not the first date the equity could trade on any exchange. Pre-2021-05-07 prices are not inherently pre-listing. The 2020-10-05 first bar is chronologically consistent with BSE listing. NSE permitted-to-trade status is documented, but the first PTT date is not established by the cited letter.

**Disposition:** Classify as MULTI_VENUE_AND_NSE_PTT; do not move the BSE listing date to 2021. Do not assert that any individual pre-May-2021 bar is NSE-sourced. Obtain the original BSE listing notice, dated BSE masters, and dated NSE permitted-category/security records. Keep identity interval approval pending those captures.

Sources:
- https://www.camsonline.com/Annual_Report/FY2020-2021/key-business-highlights.html
- https://www.camsonline.com/assets/PDF/BSE_Intimations/letter_to_BSE_on_NSE_Listing_060521.pdf
- https://www.camsonline.com/assets/PDF/CAMS-Press-Release-NSE-Listing-Trading-Final.pdf

### 2. DALBHARAT — restructuring with two distinct listing/trading permission dates

**Official facts**
- Dalmia Bharat's FY2018-19 annual report says NSE and BSE granted listing/trading permission effective 2018-12-21.
- It distinguishes shares issued to former OCL India shareholders, which began trading on 2018-12-21, from a later issue/allotment to former Dalmia Bharat Limited shareholders, for which permissions took effect 2019-01-25.
- The current NSE listing master carries 2019-01-22, which falls between those two documented dates.
- Current issuer information identifies symbol DALBHARAT, BSE code 542216 and ISIN INE00R701025.

**Interpretation:** The one-date NSE master value does not describe the full restructuring/listing event. The dates may refer to different allotment blocks and must not be collapsed into one identity interval. The annual report supports a 2018-12-21 trading commencement for one block, but the precise mapping of each block to ISIN, symbol/series, and exchange master rows must be recovered before joining prices.

**Disposition:** Reconcile 2018-12-21 and 2019-01-25 permissions against original NSE/BSE notices and dated security masters. Keep the current ISIN and any predecessor OCL identity separate unless original exchange records prove continuity. Do not automatically treat all pre-2019-01-22 DALBHARAT bars as the same current ISIN.

Sources:
- https://dalmiabharat.com/wp-content/uploads/2020/02/Annual-Report_2018-2019.pdf
- https://www.dalmiacement.com/investors-relations/shareholders-Information

### 3. FORCEMOT — BSE-listed before NSE admission

**Official facts**
- A BSE-published 2013 market document records FORCE MOTR under BSE code 500033.
- A 2025 issuer filing to BSE and NSE identifies BSE code 500033, NSE symbol FORCEMOT and ISIN INE451A01017.
- Force Motors' investor archive lists its Regulation 30 notice for NSE listing dated 2024-02-13; NSE admission took effect 2024-02-14.
- The checked OHLCV history begins 2019-08-19.

**Interpretation:** 2024-02-14 is the NSE admission date. It cannot be used as the company's first-ever exchange listing date because BSE trading under code 500033 is evidenced years earlier. The 2019 OHLCV start is not inherently pre-listing at the company/venue level. The actual venue of each bar remains unknown.

**Disposition:** Store BSE and NSE identity intervals separately. Acquire dated BSE masters for 2019 and the original NSE admission circular/dated NSE master for 2024. Do not rewrite BSE history as NSE history and do not infer that historical vendor bars came from BSE solely because BSE was available.

Sources:
- https://www.bseindia.com/downloads1/Underlying_Disclosurer_under_LEIPS_Cash_Oct.pdf
- https://www.forcemotors.com/investor/
- https://www.bseindia.com/xml-data/corpfiling/AttachHis/9f038e36-efde-4824-ad33-4e30bd79f6f0.pdf
- https://www.nseindia.in/get-quote/equity/FORCEMOT/Forcemot-Ns

### 4. NESTLEIND — long-standing BSE history and NSE permitted-to-trade before formal listing

**Official facts**
- Nestlé India's FY2023-24 annual report states that its equity was listed on BSE and allowed to trade on NSE under the Permitted to Trade category in cash and F&O segments.
- The same report states that formal NSE listing took effect 2023-08-01.
- Nestlé's exchange intimation says the shares were already allowed to trade on NSE under the PTT category before formal listing.
- The checked OHLCV history begins 2018-01-01.

**Interpretation:** 2023-08-01 is the formal NSE listing date, not the first date of BSE listing or NSE trading eligibility. Historical rows are not inherently pre-listing at the issuer level. The exact first NSE PTT date and source venue of each price row are not established by the annual report/intimation.

**Disposition:** Maintain separate BSE-listed and NSE-PTT/formal-NSE states. Obtain historical BSE and NSE permitted-category/security masters and the original listing letter. Do not backdate formal NSE listing to the first OHLCV date.

Sources:
- https://nsearchives.nseindia.com/annual_reports/AR_24138_NESTLEIND_2023_2024_15062024233329.pdf
- https://www.nestle.in/media/specialannouncements/regulation-30-sebi-listing-obligations-disclosure-requirements-regulations-2015-%E2%80%9Clisting
- https://www.nseindia.in/get-quote/equity/NESTLEIND/Nestl-India

### 5. PATANJALI — RUCHI predecessor, suspension/recommencement after capital reduction, then symbol change

**Official facts**
- NSE circular CML43307 dated 2020-01-22 states that trading in Ruchi Soya equity recommenced on NSE effective 2020-01-24 after capital reduction pursuant to the NCLT-approved resolution plan.
- NSE circular CML52890 states that the name and trading symbol changed from RUCHI / Ruchi Soya Industries Limited to PATANJALI / Patanjali Foods Limited effective 2022-07-13.
- The 2018-19 issuer annual report describes the CIRP and capital restructuring. NSE's 2020 circular is explicit that this is recommencement after capital reduction, not an ordinary uninterrupted listing.
- The checked OHLCV history begins 2018-01-01, before the 2020 recommencement date.

**Interpretation:** Pre-2020 observations belong to the pre-recommencement Ruchi Soya trading history if genuine. They must not be treated as an uninterrupted price history of the post-capital-reduction security without old/new ISIN, share-capital reduction/consolidation, and adjustment methodology being established. The RUCHI-to-PATANJALI symbol change in 2022 is a separate event and does not by itself prove ISIN continuity.

**Disposition:** Split the timeline into (a) pre-suspension/pre-recommencement RUCHI security, (b) post-2020 recommenced RUCHI security after capital reduction, and (c) PATANJALI symbol/name from 2022-07-13. Obtain exact NSE/BSE suspension, delisting/recommencement, old/new ISIN and dated master records, plus corporate-action terms. No pre-2020 bars may enter the current-security series until relationship and adjustment are independently reviewed.

Sources:
- https://archives.nseindia.com/content/circulars/CML43307.pdf
- https://archives.nseindia.com/content/circulars/CML52890.pdf
- https://www.bseindia.com/bseplus/AnnualReport/500368/5003680319.pdf
- https://www.nseindia.com/companies-listing/corporate-filings-actions

### 6. TECHNOE — predecessor TECHNOELEC suspended; successor shares listed under a new identity

**Official facts**
- NSE's 2007-03-29 press release says TECHNOELEC, ISIN INE470B01023, was listed and admitted to dealings effective 2007-04-02.
- Techno Electric's 2018 amalgamation materials state that trading in the old shares was suspended from 2018-08-08 and new shares in the merged entity were listed on BSE and NSE on 2018-12-04.
- NSE's 2018-11-30 listing approval says the new equity shares allotted pursuant to the scheme would be listed effective 2018-12-04 under symbol TECHNOE.
- The current security is identified as TECHNOE / ISIN INE285K01026.
- The checked OHLCV history begins 2018-01-01.

**Interpretation:** Pre-2018-08-08 history is predecessor-security history, not automatically the same security as post-2018-12-04 TECHNOE shares. 2018-08-08 through 2018-12-03 is a documented suspension/no-dealings interval for the old shares. The new issue has a different ISIN. Any continuous vendor series across this event must be tested for corporate-action adjustment and source ticker remapping.

**Disposition:** Establish separate intervals for TECHNOELEC / INE470B01023 ending no later than 2018-08-07 and TECHNOE / INE285K01026 beginning 2018-12-04, subject to capture of original exchange notices and dated masters. Preserve the suspension interval. Do not stitch prices across the ISIN change without a separately approved successor-security/corporate-action mapping.

Sources:
- https://nsearchives.nseindia.com/content/press/29032007.htm
- https://www.techno.co.in/public/uploads/2/2019-12/listing_cum_trading_approval_letter_nse.pdf
- https://www.techno.co.in/public/investor/amalgamation
- https://www.techno.co.in/public/uploads/2/2019-12/letter_of_offer_buyback.pdf
- https://www.techno.co.in/investor/securities/share/listing_details

## What is resolved versus what is not

| Case | Event interpretation | Per-row source venue | Point-in-time identity interval |
|---|---|---|---|
| CAMS | Explained by BSE-first and NSE PTT/formal listing sequence | Not established | Pending dated masters and PTT start |
| DALBHARAT | Explained by two restructuring allotment/listing windows | Not established | Pending per-block ISIN/master reconciliation |
| FORCEMOT | Explained by BSE history predating NSE admission | Not established | Pending dated BSE/NSE masters |
| NESTLEIND | Explained by BSE listing and NSE PTT before formal listing | Not established | Pending dated masters and PTT start |
| PATANJALI | Explained by RUCHI predecessor and post-capital-reduction recommencement, then rename | Not established | Pending old/new ISIN and suspension/master evidence |
| TECHNOE | Explained by predecessor suspension and new successor issue/listing | Not established | Pending captured original masters and independent review |

## Required evidence before marking identity rows independently verified

1. Acquire original, dated NSE and BSE security-master files bracketing each effective date. A current master, search result, issuer presentation, or source catalogue is not a historical master snapshot.
2. Acquire original listing, suspension, permitted-to-trade, scheme, capital-reduction and symbol-change circulars referenced above.
3. Preserve exact original bytes; record canonical and retrieval URLs, UTC retrieval time, MIME type, byte length, SHA-256, archive path, and page/table/row locator in historical_source_documents.csv.
4. Add source-registry entries as DISCOVERY_ONLY until source authority and scope have been reviewed. Public URLs are discovery references, not archived evidence.
5. Populate security_identity_history.csv only from retained documents and dated masters. Use separate NSE/BSE intervals; retain historical symbol, series, ISIN and exchange security code exactly as published.
6. Have a second reviewer inspect retained bytes and approve each interval. Reviewer metadata is an attestation, not cryptographic proof.
7. Add row-level OHLCV provenance (provider, vendor ticker, source venue, retrieval batch/document) or acquire equivalent vendor evidence. Without it, historical price rows cannot be assigned to an exchange.
8. For PATANJALI, DALBHARAT and TECHNOE, complete corporate-reorganisation/adjustment review before stitching predecessor and successor prices.
9. Keep pre-event bars quarantined from the current-security training series until all identity and adjustment checks pass. Do not delete or rewrite source OHLCV.

## Current conclusion

- Six anomaly explanations have been investigated against official exchange/issuer sources.
- The single-NSE-date interpretation is invalid for these cases and must be replaced with venue-specific event/identity history.
- No source bytes have yet been archived and hashed by this investigation.
- No independently reviewed identity intervals have been created; the identity-history template must remain empty until capture and review.
- No OHLCV row has been attributed to a venue or modified.
- The overall research-readiness gate must remain blocked. These findings do not authorize training or production use.
