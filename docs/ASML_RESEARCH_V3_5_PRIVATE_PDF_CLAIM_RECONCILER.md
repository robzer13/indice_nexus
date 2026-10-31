# ASML Research v3.5 — private PDF claim reconciler (SHADOW)

Run: a1e59126-1ca9-4e55-8764-54a1e3d4dc24  
Data cutoff: 2026-10-30  
Stage: Research — no admission / no transition  
Base: v3.4 SHA-256 independent re-audit, \`scan_pages\` endpoint

## Execution
- Visit \`/admin/research-primary-sources/reconcile\` on the **isolated SHADOW Preview**, signed in as administrator.
- Click **Balayer les 3 PDF**. The browser serially calls the existing read-only \`/admin/research-primary-sources/audit/run\` endpoint with at most 20 PDF pages per call.
- Each call re-downloads the original private PDF and its receipt, recalculates SHA-256, confirms size, receipt content, PDF structure, and checks the pinned source identity before text extraction.
- The browser checks the result's source ID, SHA-256, integrity status, receipt fingerprint, exact page ranges, page-hash formatting, claim allowlist, and explicit non-admission flags. It fails closed.
- Stop and resume within the same open browser tab. Refreshing/closing discards the in-memory partial scan. The user may export the partial report before closing.
- After the entire scan (or when interrupted), **Exporter le JSON de réconciliation** creates a local file; the server never persists the matrix.

## Scope
Exactly 3 previously archived private PDFs, with 19 linked evidence claim references:
- ASML FY2025 US GAAP: 15, including 2 derived claims requiring separate calculation (C-003, C-009).
- ASML Q3 2026 US GAAP: 2.
- TSMC Q3 2026 edited call transcript: 2.

The client matrix preserves \`CLAIM_ID\`, \`SOURCE_SHA256\`, PDF page (1-based), page text hash, numeric probe, source excerpt, context-hint observation, and evidence-acceptance flags.

## Semantics
- \`NUMERIC_CANDIDATES_PRESENT\`: at least one number detected by the v3.4 scanner, not a claim validation.
- \`HINT_PRESENT\` / \`HINT_ABSENT\`: heuristic substring check inside the excerpt, never proof or disproof.
- \`NO_NUMERIC_MATCH_IN_EXTRACTED_TEXT\`: no match in text extracted across a complete PDF, not proof the fact is absent from the PDF; text extraction and single-hit-per-page truncation may miss information.
- \`DERIVED_RECONCILIATION_REQUIRED\`: derived values are not searched as if independently reported.
- \`PARTIAL_SCAN\`: pages remain unexamined.
- \`validatedClaimCount=0\`, \`registryAdmitted=false\` are invariant.

The v3.4 scanner currently returns at most one lexical match per claim per page and at most 12 matches per page. Therefore the matrix is an **automated locator**, not an exhaustive semantic verifier. A separate human page review, cross-source checks and accounting reconciliation are necessary before any evidence-level decision.

## Security and promotion boundary
The page uses the existing administrator session, SHADOW host gate, strict same-origin POST policy, private Supabase Storage reads, and no write endpoint. The v3.5 front-end does not call the Registry or any production API.

No change to the frozen methodology, claim values, conflict ledger, certification, scoring, data cutoff, stage gate, run registry or canonical snapshot is authorized by this version.

## Required acceptance checks
1. Test suite / TypeScript / lint / Next build.
2. Vercel Preview target is the isolated SHADOW project, not production.
3. Verify matrix covers 3 sources, 19 claims, and reports all derived claim statuses.
4. Verify page 289 C-010/C-011 are located as numeric candidates, and page 295 C-012 is not falsely validated as China exposure.
5. Export JSON, verify source hashes and continuity of page coverage.
6. No Registry artifact admitted; no Deep Dive handoff or publication.
