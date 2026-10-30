# OroTitan ASML — Official-source PDF capture, SHADOW v2.9

**RUN_ID** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24`  
**DATA_CUTOFF** `2026-10-30`  
**Primary process** `OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1`  
**Research contract** `OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0`  
**Classification** `PRODUCT_SHADOW_CAPTURE_PRE_ADMISSION_ONLY`  
**No Registry mutation / no deployment / no staging promotion in this work package.**

## Outcome

A read-only research prototype plus a private SHADOW capture implementation were produced locally. The downloader is **not deployed**. The environmental attempts to retrieve original bytes were unsuccessful; **zero actual issuer PDFs were captured, SHA-sealed or Registry-admitted in this turn**. The unit tests use a synthetic PDF whose contents have no research authority.

Three exact document IDs are accepted; the browser cannot provide an arbitrary URL:

| Document ID | Root URL | Publication basis | Accounting / provenance basis |
|---|---|---|---|
| `ASML_2025_US_GAAP` | `https://ourbrand.asml.com/m/71076aaad607de4d/original/asml-2025-annual-report-based-on-us-gaap.pdf` | 2026-02-25 | US GAAP 2025, full issuer PDF |
| `ASML_2026_Q3_US_GAAP` | `https://www.investor.asml.com/static-files/d310262b-d037-485d-83b5-92c93cc462f0` | 2026-10-14 | US GAAP, unaudited Q3 2026 |
| `TSMC_2026_Q3_CALL` | `https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2026-10/eba4fac72873bf546b669bd2b5d973df5c93215d/TSMC%203Q26%20Transcript.pdf` | 2026-10-15 | Issuer-hosted **LSEG-edited transcript**, not authenticated call audio |

ASML's annual PDF official download may return a **302 within `ourbrand.asml.com`** to `/m/71076aaad607de4d/original/Exhibit%252015.2.pdf`; this specific origin/prefix is permitted. Any cross-origin or non-HTTPS redirect fails closed. If the Q3 static asset or TSMC transcript redirects to a third-party CDN, **do not relax the allowlist without independently vetting the actual target origin**.

## Files supplied

1. `lib/orotitan-equity/product/asml-official-pdf-capture-v29.ts` — exact allowlist; bounded HTTPS GET, redirect validation, PDF header/EOF check; SHA-256 and private Supabase readback; immutable create-only source and receipt.
2. `app/admin/research-primary-sources/capture/route.ts` — server-only endpoint. Requires exact existing `isHostedShadowResearchEnabled` gate, an authenticated admin session, same-origin POST, and short form-only request.
3. `app/admin/research-primary-sources/page.tsx` — restricted UI showing the three source IDs and receipt integrity outcome.
4. `tests/asml-official-pdf-capture-v29.test.mjs` — synthetic fixtures for positive capture, retries, safe redirect, off-domain redirect rejection, public bucket denial, invalid file/MIME/length, and tampered storage readback.

Do **not** replace the existing Q3 browser-upload route with this module; it is an additional path. The existing Q3 upload is already present in SHADOW and the pre-existing SHA is `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9` (application-reported, not independently reacquired here). The new receipt contains a Boolean comparison if the newly downloaded Q3 bytes match this previous SHA.

## Security and governance controls

- **SHADOW-only**: existing environment gate requires Vercel Preview, the exact SHADOW Supabase project `awgsurdyvsyolcgpnygh`, matching non-public credentials and admin session. No production endpoint.
- **No arbitrary URL** or domain supplied in the POST. Exactly three hardcoded document IDs.
- **Fail-closed redirect policy**: HTTPS, exact hostname, path prefix, empty credentials, empty query string, maximum two redirects.
- **Bounded streaming**: 15–40 MB per source, check announced content length and actual bytes, 45s source fetch timeout, request body <= 2,000 bytes. The Next handler declares `maxDuration=60`, subject to hosting plan restrictions.
- **No stored file served publicly**: existing bucket `orotitan-source-files-v1` must be private (`public === false`) before any fetch/storage. Writes are create-only, no upsert, followed by download and exact SHA verification.
- **Idempotency**: content-addressed paths, duplicate requires the same bytes, receipt preserves first capture timestamp and validates fields on retry.
- **Evidence boundary**: HTTPS transport establishes the hostname presented to runtime TLS; it **does not** establish a digitally signed ASML/TSMC original or prove that the particular PDF bytes were unchanged at historical cutoff. Receipt fields `sourceOriginCryptographicallyAttested=false`, `documentAuthenticityIndependentlyVerified=false`, `historicalBytesAtCutoffIndependentlyVerified=false` remain false. No claim of independent source-origin certification.
- **Registry firewall**: code has no calls to `orotitan_runs`, `orotitan_run_stages`, `orotitan_artifacts`, no import of writers, no stage mutation, no production publication and no scoring/valuation.

## Local verification

Run from the patch root with Node.js 22:

```bash
node --experimental-strip-types --test tests/asml-official-pdf-capture-v29.test.mjs
```

Observed result in this preparation: **8 tests passed, 0 failed**. This is unit testing using synthetic PDF bytes and mock private storage; not a Vercel integration test, nor a real upstream/TLS run. A standalone syntax/TSX parse check was also performed. Full Next.js typecheck, lint and deployed end-to-end tests are **not yet executed** against the repository dependency tree.

## Integration / live-capture protocol

1. Apply patch **to a dedicated branch** rooted in `feat/orotitan-product-recovery-loop-shadow-20261030`, without merging draft PR #436 or modifying production.
2. Install dependencies already pinned in the repository, run `npm run typecheck`, `npm run lint`, `npm test` and review Next route compatibility before deploying.
3. Deploy **only** to the Vercel Preview bound to the isolated SHADOW Supabase ref `awgsurdyvsyolcgpnygh`. Verify both standard app and research intake clients target that ref. Never enable against production `cugpgtzygqqlxetyeven`.
4. Authenticate as admin; open `/admin/research-primary-sources` and capture **ASML_2025_US_GAAP** first. Check the receipt `sha256`, raw byte count, final URL, redirect path, storage content hash and source-publication date.
5. Capture the TSMC edited transcript and then optionally the Q3 PDF for an independent comparison with the existing SHADOW receipt. If HTTPS or redirect policy blocks, document the exact error **privately** and use a separately reviewed browser-transmitted fallback; do not silently widen the allowlist.
6. Perform an independent readback of the receipts and original bytes. Reconcile claims and accounting bases (SEC accession `011378` = US GAAP; `011377` = EU-IFRS statutory).
7. Independently decide material admissibility under the pinned contract, not from a success message. Build authoritative Evidence Ledger lineage and exact Registry source artifacts only after required approvals and gate checks.

## Research handoff

`RUN_STATUS=ACTIVE`; `STAGE=RESEARCH`; `READY_FOR_DEEP_DIVE=NOT_EVALUATED`; `REGISTERED_ARTIFACTS=0` as last read. Remains working/nonadmitted. Economic gaps remain: G001 exact point-in-time share count, G006 official price reference, incremental ROIC, maintenance CapEx, and net factoring cash timing effect pending ASML IR.
