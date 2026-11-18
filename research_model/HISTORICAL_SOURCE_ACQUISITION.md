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

When supplied, the source registry is used to reject document rows whose source ID is not registered. The validator rejects duplicate document IDs, malformed HTTPS URLs, invalid dates/UTC timestamps, reversed effective intervals, unsafe paths, missing files, byte-length mismatches, and SHA-256 mismatches. It does not fetch URLs or modify source documents. CI uses only the empty template and deterministic synthetic fixtures; it must not download exchange data.

## Evidence and approval limits

- A matching SHA-256 proves byte integrity relative to the manifest, not that the source is genuine, complete, or authoritative.
- A reviewed source registry entry does not prove that a particular archived document supports a particular identity interval.
- A reviewed document does not itself validate an ISIN/symbol/series mapping. Every mapping needs a precise document locator, source-specific date semantics, and independent human review.
- Do not infer effective dates from file names, retrieval dates, or the first/last observation in a dataset.
- Preserve conflicting documents and record the conflict; do not silently choose one.
- Do not merge different ISINs or infer successor/predecessor relationships from symbol similarity.
- No identity mapping, corporate-action event, point-in-time universe membership, or historical partition is authorized by this acquisition contract.

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
