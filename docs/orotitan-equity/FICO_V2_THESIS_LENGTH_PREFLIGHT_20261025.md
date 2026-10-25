# FICO — V2.0.11 Thesis Compatibility Admission Preflight

Date: 2026-10-25
Scope: existing FICO run only, immutable source checks and V2.0.11 review-candidate code.
**State:** PASS for the *read-only preparation conditions*; **not** an approved production runtime activation or a publisher command.
**Evidence provenance:** queried production Registry read-only; fetched exact immutable private `real-orotitan` Git blobs; separately inspected PR #425 and CI.

## 1. Existing run identity and status — PRODUCTION READ ONLY

| Field | Verified value |
|---|---|
| RUN_ID | `796edd3d-2249-4f1b-a47b-2404edb92097` |
| RUN_STATUS / CURRENT_STAGE | `ACTIVE / INTEGRATION` |
| Run state version | `11` |
| DATA_CUTOFF | `2026-09-19` |
| Frozen V2 Contract Set SHA-256 | `1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e` |
| Issuer | `b90b4161-6010-4c5f-99b1-1e357833c10e` |
| Security | `202d4e98-a2d5-461c-bc41-eade04e5e646` |
| Dossier | `40cb8e23-6e4d-4fb0-a565-66c292d57c05` |
| Current canonical snapshot | `NULL` |
| Existing historical snapshots for this dossier | `0` |
| Publication authorization / success events | `0 / 0` |

Stages:
- RESEARCH: `COMPLETE / FINAL / YES`.
- DEEP_DIVE: `COMPLETE / FINAL / YES`; active manifest `3d69ad25-c939-4cf7-8b9d-df74eb1d64f3@1`; state version 6, no critical blockers.
- INTEGRATION: `IN_PROGRESS`, stage state version 1, no active manifest, `NOT_EVALUATED` handoff gate, **zero** Integration artifacts.

Therefore `resume_orotitan_stage` is **not applicable** (run is not PAUSED), restarting the stage with `start_orotitan_stage` is **not applicable** (already started), and a second run must **not** be created. The correct proposed route is to continue the existing empty IN_PROGRESS Integration stage once compatibility is admitted and Integration outputs are actually built.

## 2. Exact private source provenance — PASS

Private SHA-1 Git source commit: `c878b707ac54f70f4803c528a3cf4394cf87619a`. No sealed source is rewritten or published into `indice_nexus`.

| Existing artifact | Registered SHA-256 verified from bytes | Registered byte size verified |
|---|---|---:|
| `STRUCTURED_INVESTMENT_THESIS_ARTIFACT b9e78408-447e-4de0-8eb6-58680782879f@1` | `805c6991105fdd02092a5d3e0fb28e536bc75645e4c053bf7d7151b28b44dfee` | 1,668 |
| `DEEP_DIVE_REPORT bd9e7b3d-5ad3-483b-84ec-8f5f314a67c7@1` | `727355112ca5bd82de077a12b648e3bae06d97a6bdb6a6653ec2eecde4715f29` | 7,392 |
| `FINAL Deep Dive Manifest 3d69ad25-c939-4cf7-8b9d-df74eb1d64f3@1` | `2fbacdd0b37897ba24348a8517cb4e3b1cc83b44bda4269b862a3af30781e57a` | 26,604 |

Source-to-report cross-check: `QUALITY_CASE`, `VALUATION_CASE` and `KEY_RISK` match **exactly**, with lengths 385 / 284 / 217. All three source artifacts have the exact run ID and `2026-09-19` cutoff. The registered Git blob hashes and the byte counts also matched. Independent SHA-256 implementation test vector `SHA256("abc")` was checked before byte-level comparisons.

## 3. V2.0.11 compatibility design — PASS on review branch

- Additive overlay relaxes **only** maxLength for the three structured thesis fields.
- Legacy V2 `04_SCREENER_SCHEMA_V2.json`, V2.0.10 validator bytes, V1 frozen schema and all V2 contract pins remain **unchanged**.
- Opt-in composed validator still checks V1 canonical core, V2 classification/description semantics, deterministic reconciliations and the independent I3-B boundary when exercised during integration.
- Two-source cryptographic admission guard requires unchanged UTF-8 source bytes and identical exact fields in the frozen `STRUCTURED_INVESTMENT_THESIS_ARTIFACT`, `DEEP_DIVE_REPORT` and proposed canonical candidate. Rejects tampered source hashes, mismatched issuer/security/cutoff, rewrite or divergent report.
- Preflight run-state function adds narrowly scoped `EXISTING_INTEGRATION_EMPTY_IN_PROGRESS`, **read-only only**, fail-closed if any Integration artifacts, changed stage version, existing snapshot, publication event, contract mutation or missing versioned test evidence.
- None of these functions registers artifacts, changes Registry stage status, writes canonical snapshots or grants GO PUBLISH.

CI review code head: `93be59157fb0bfa7cb2a7e22a5797094d21d0432`.
GitHub Actions `Screener CI` run `40177241289`: **SUCCESS** (lint, typecheck, unit tests including provenance/negative matrix, disposable PostgreSQL 17 tests and build).

## 4. Remaining gates — NOT YET SATISFIED

1. Ratify the exact V2.0.11 emergency compatibility patch with controlled, immutable version/hash identity, independent review and no freeze bypass.
2. Controlled Shadow execution of the **full** FICO Integration candidate, using the actual complete Deep Dive certified sources (not synthetic tests) and exact V1-core+V2 compatibility validation. Do not copy private source bodies to public reviews.
3. Generate and persist eight mandatory Integration artifacts after validating: `CANONICAL_SNAPSHOT_CANDIDATE`, `INTEGRATION_MAPPING_RECORD`, `SCHEMA_VALIDATION_REPORT`, `I2_RECONCILIATION_REPORT`, `HISTORY_TRANSITION_VALIDATION_REPORT`, `I3B_ADMISSION_REPORT`, `PRE_PUBLICATION_CONTROL_CARD`, `INTEGRATION_STAGE_MANIFEST`.
4. Verify all eight immutable SHA-256 bytes and Registry relationships; finalize Integration with the exact authorized Registry RPC and idempotence guards. Only then may FICO reach `READY_TO_PUBLISH`.
5. Require **distinct explicit** `GO PUBLISH FICO` authorization before canonical snapshot promotion. Generic `ok go` for compatibility work is not GO PUBLISH.

## Decision

- **Admitted for further review / preflight preparation:** YES, no identified need to shorten certified FICO prose.
- **Runtime patch activated / deployed to production:** NO.
- **FICO Integration completed:** NO.
- **FICO published:** NO.
- **Frozen contracts or existing artifacts changed:** NO.
