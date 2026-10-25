# OroTitan V2 Thesis Length Compatibility Patch V2.0.11

**Status:** REVIEW CANDIDATE — NOT ADMITTED OR DEPLOYED  
**Class:** blocking canonical-output representability defect / frozen-contract contradiction  
**Methodology, scoring, I2, I3-B and valuation:** UNCHANGED  
**Changes to frozen V1 or V2 bytes / existing Contract Pin Pack:** NONE

## 1. Observed, generalizable defect

The frozen V2 Deep Dive Stage Contract §5.5 states each certified
`QUALITY_CASE`, `VALUATION_CASE`, and `KEY_RISK` **should normally** be
at most 240 characters. This is a presentation preference, not a hard
non-admission rule.

However, `04_SCREENER_SCHEMA_V2.json` and
`validateV2ProductSemantics` impose **hard 240-character ceilings**, so
an otherwise admitted, traceable, immutable and accurately certified
thesis of 241+ characters cannot be mapped into the canonical V2 output
without either illicit truncation or reopening analytical certification.

Observed on existing FICO run
`796edd3d-2249-4f1b-a47b-2404edb92097`:
`quality_case` 385 characters, `valuation_case` 284 characters,
`key_risk` 217 characters. The original source artifacts are in private
`real-orotitan`, must remain there and remain immutable. No certified
text is reproduced in this public review.

This is an independently repeatable compatibility defect, not a
company-specific waiver or a preference for long UI copy.

## 2. Exact, minimal additive correction

New immutable candidate overlay:
`04_SCREENER_SCHEMA_V2_THESIS_COMPAT_V2.0.11.json`.

Only the three `maxLength: 240` rules for
`v2_product.investment_thesis.quality_case`, `valuation_case`,
`key_risk` are removed in the **additive compatibility overlay**.
`type: string`, `minLength: 1`, required fields and
`additionalProperties: false` are preserved. All other V2 product
rules, ISO countries, taxonomy, business summary limits and V1 core
schema are unchanged.

There is no replacement arbitrary maximum: exact certified prose can
be longer than a preview display. The compact user interface may show
about 650 Unicode characters with an expand action, without changing
the stored source. A pure preview helper is provided; this change
does **not** deploy a new interactive UI or silently truncate snapshots.

New opt-in validation / persistence adapter:
`lib/orotitan-equity/v2/research-snapshot-schema-thesis-compat.ts`
and `persistence-core-thesis-compat.ts`. Both preserve the exact input
JSON object and continue invoking the unchanged V1-core composed
validator and unchanged V2 persistence RPC only **after** independent
admission. The legacy validator and legacy writer are not rewired.
The frozen V2.0.10 validator-hash acceptance test remains valid.

## 3. Compatibility governance and proof before runtime use

Under frozen V2 Process §15, this patch is **not** admissible solely
because an individual user dislikes a limit. It requires an operator
to certify the demonstrated blocking upstream/canonical inconsistency,
the narrowness of the remedy and PASS for all regressions.

For an existing run, all of the following must be rechecked:

1. Exact run, issuer, security, dossier, `DATA_CUTOFF` and frozen 13-pin
   contract set; **no rebind** of any pin or source.
2. Deep Dive `COMPLETE`, `FINAL`, `READY_FOR_INTEGRATION=YES`,
   exact certified thesis source IDs/versions/bytes/SHA-256 and
   absence of an analytical contradiction.
3. Every new canonical thesis field equals the certified source field
   **byte-for-byte**; no summarization or truncation.
4. No admitted Integration candidate, no `READY_TO_PUBLISH`,
   no current pointer promotion and no publication authorization.
5. Exact additive overlay and candidate code SHA-256 recorded;
   frozen V1/V2 artifacts and V2.0.10 validator unchanged.
6. Frozen core + additive overlay validation, I2 reconciliation, I3-B
   admission, semantic state integrity, history transitions, final
   manifest and artifact persistence all **PASS**.
7. The resulting Integration artifacts use the existing Registry RPCs,
   durable immutable artifact bodies and idempotent transitions.
8. A separate `GO PUBLISH <COMPANY>` must occur after complete
   Integration — **not** inferred from development approval.

For FICO, Integration is already `IN_PROGRESS` with no Integration
artifacts yet. This patch supplies no permission for a stage rollback,
a Deep Dive reopen, an arbitrary direct SQL edit, a stage-finalization
shortcut or publication. The actual production state must be
independently re-read at admission time.

## 4. Immutable history and traceability

**Never modify** these files:

- `contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2.json`;
- `contracts/orotitan-equity/v2/OROTITAN_CANONICAL_PRODUCT_EXTENSION_V2.schema.json`;
- `contracts/orotitan-equity/v2/04_INTEGRATION_SPEC_V2.md`;
- `lib/orotitan-equity/v2/research-snapshot-schema.ts`;
- `lib/orotitan-equity/v2/product.ts`;
- `lib/orotitan-equity/v2/existing-run-compatibility-admission.ts`.

No database migration, production deploy, publisher action, `GO`, or
Registry mutation is part of this code review.

## 5. Required regression evidence

- 385 / 284 / 1100-character synthetic inputs accepted, byte-exact.
- Frozen V2 validator continues to reject them; its pinned hash unchanged.
- Short V2 snapshots work identically.
- Only the three thesis maxLength schema properties differ.
- Empty strings, wrong field types, unknown keys, invalid country,
  long business description and missing V1 core still reject.
- UI preview returns unmodified full text separately.
- Mocked writer invokes the exact unchanged V2 RPC payload only after
  the additive validator passes. No live database write in tests.
- Existing lint, TypeScript, unit tests, disposable PostgreSQL tests and
  `next build` pass in CI.

**Admission status:** PENDING independent CI and operator review.
