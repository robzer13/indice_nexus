# OroTitan — L'Oreal pilot 2: Research portability V1 (SHADOW)
Date: 2026-10-31 | Historical data cutoff: 2026-10-30

## Authority and scope

- PILOTAGE_PRE_FLIGHT only. L'Oreal RUN_ID = NOT_PERSISTED.
- This commit is a **software portability experiment**, not Research-stage admission.
- No production mutation, no SHADOW Registry write, no primary PDF upload,
  no Evidence Ledger admission, no scoring, no Deep Dive transition.
- Phase-0 inputs come from official issuer HTML manually transcribed into
  `LOREAL_PILOT2_2026-10-30_PRE_ADMISSION_SOURCE_AND_CLAIM_CANDIDATES_V0_1.json`.
- Financial facts and formulas remain **candidates** until source-by-source
  claim validation and exact issuer-PDF provenance are established.

## Implemented

1. Generic numeric PDF triage:
   `lib/orotitan-equity/product/research-pdf-numeric-core-v1.ts`
   - Configurable claim ID, probe and optional context alternatives.
   - First match per probe per PDF page; no false claim validation.
   - Old ASML v3.4 uses the shared core through its original public function.
   - No private PDF path, receipt, issuer, or cutoff restrictions relaxed.

2. Generic exact-decimal financial reconciliation:
   `lib/orotitan-equity/product/research-financial-arithmetic-qa-v1.ts`
   - Uses BigInt to avoid binary floating error.
   - Auto-derived maximum rounding variance:
     `(number_of_terms + 1) * 0.5 * (unit_of_last_reported_decimal)`.
   - Preserves EXACT_MATCH / ROUNDING_COMPATIBLE /
     OUT_OF_TOLERANCE without producing evidence authority.

3. Fail-closed source eligibility:
   `lib/orotitan-equity/product/research-private-source-readiness-v1.ts`
   - Requires exact original bytes hash, byte size, private storage reference,
     receipt hash, source date, capture time and independently verified state.
   - Explicit post-cutoff publication handling and historical-byte caution.
   - This **is not** a capture or receipt-verification implementation.

4. L'Oreal isolated configuration, with two as-yet unsealed PDF sources:
   `lib/orotitan-equity/product/loreal-pilot2-candidate-config-v1.ts`
   - 8 source-specific PDF numeric probes.
   - 7 accounting QA checks on openly published issuer HTML.
   - Financial interpretation: issuer-defined cash flow is not IFRS CFO/FCF.

5. Admin-only / SHADOW-only read-only preflight dashboard:
   `/admin/research-portability/loreal`.
   - Explicit missing byte/receipt status and stage admission block.

6. Test suite: `tests/research-loreal-portability-v1.test.ts`.

## Seven financial reconciliations

The source-only candidate input set is from L'Oreal official issuer releases.
All monetary amounts EUR million.

| Check | Calculated | Published | Difference | Auto tolerance | Diagnostic |
|---|---:|---:|---:|---:|---|
| 9M26 divisions | 35,337.6 | 35,337.6 | 0.0 | 0.25 | EXACT |
| 9M26 regions | 35,337.6 | 35,337.6 | 0.0 | 0.30 | EXACT |
| H1-26 divisions | 23,776.1 | 23,776.1 | 0.0 | 0.25 | EXACT |
| Q3-26 divisions | 11,561.6 | 11,561.4 | 0.2 | 0.25 | ROUNDING |
| Q3-26 regions | 11,561.5 | 11,561.4 | 0.1 | 0.30 | ROUNDING |
| H1+Q3 versus 9M26 | 35,337.5 | 35,337.6 | -0.1 | 0.15 | ROUNDING |
| 9M25 + Q4-25 vs FY25 | 44,052.0 | 44,052.0 | 0.0 | 0.15 | EXACT |

These are mathematical reconciliations of unsealed transcriptions, not
independent validation of figures or disclosure semantics.

Issuer pages:
- https://www.loreal-finance.com/eng/press-release/2025-annual-results
- https://www.loreal-finance.com/eng/press-release/2026-half-year-results
- https://www.loreal-finance.com/eng/press-release/sales-30-september-2026

## Run creation blocker: LR-G-REG-01

The existing SHADOW registry includes example V2-rc1 contract-pinned runs.
The frozen V1 execution chain supplied in project uploads cannot be silently
replaced by those V2-rc1 examples.

Registry requires 13 complete and immutable contract pin entries. The
submitted local file set is insufficient to resolve the original V1
investment_policy, execution_patch, i2 and i3b contracts with valid exact
hashes/locators. Some may exist in another private archive, but they have
not been independently resolved for this run.

FAIL CLOSED: no `orotitan_runs` row, issuer identity insert or stage insert
until a complete consistent chain is resolved. Do not fabricate pin hashes,
private storage URIs, or compare a V1 scope to V2-rc1 as equivalent.

## Not yet implemented

- Generic **private** PDF capture to Supabase Storage with reliable original
  byte source, content SHA, immutable independently verified receipt.
- Generic server endpoint to rehash stored bytes and run the shared probes.
- Root-origin publisher authentication and exact historical byte proof.
- IFRS normalized CFO, owner earnings and ROIC, cutoff shares/price.
- A compliant persistent Research run or artifact registry entries.

## Next acceptance conditions

A. Resolve the 13 contract pins for one coherent version of OroTitan.
B. Create issuer/security identity and idempotent run only after A.
C. Capture original L'Oreal PDF bytes into private storage with receipts.
D. Execute generic PDF scan and compare its page evidence with public
   HTML transcriptions; perform visual checks and claim-specific reconciliation.
E. Measure how much work reused unchanged from ASML before making speed claims.
