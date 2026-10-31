# OroTitan Run Factory P0-E — first V3.1 INITIAL run (SHADOW only)

Date: 2026-10-31
Operator authority: **GO P0-E — CRÉATION DU RUN L’ORÉAL EN SHADOW — V3.1 — CUTOFF 30/10/2026**
Production project `cugpgtzygqqlxetyeven`: OUT OF SCOPE, NO WRITES.
SHADOW project `awgsurdyvsyolcgpnygh`: only authorized target.

## Scope

- Exact canonical identity: L’Oréal S.A. / FR0000120321 / XPAR
- Issuer: `c431698b-3b5f-40c4-bf24-8bf4612db876`
- Security: `44bf0acd-9420-42f3-bdd1-1e725740154d`
- Dossier: `e30bd3e4-618c-41a2-93b5-015fc2ea708e`
- Cutoff: `2026-10-30`
- Run type: `INITIAL`; no baseline snapshot; no parent.
- Frozen V3.1 analytic profile, pin pack `3.0.2`:
  `3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2`.
- Production V3.0.3 bootstrap has an intentional production environment assertion.
  **Do not lie about SHADOW being production.** The P0-E planner independently
  derives the same immutable descriptor and request fingerprint; the official
  bootstrap is used only as a disconnected deterministic parity test.
- The Registry `orotitan_generation` enum V1/V2 is a Registry schema
  generation label, **not** the analytic method V3.1. Existing frozen V3 runtime
  explicitly requires no schema migration and recognizes the analytic
  authority by `contract_pins` + `contract_set_sha256`. No migration to
  `orotitan_generation` is authorized.

## Implemented

1. `lib/orotitan-equity/product/runfactory-p0e-byte-gate.ts`:
   pinned Git object SHA-1 and canonical SHA-256 verification, all 15
   original files, bounded download, no Registry changes.
2. `lib/orotitan-equity/product/runfactory-p0e-shadow-plan-v1.ts`:
   deterministic shadow-only V3.1 fingerprint identical to frozen V3.0.3,
   proper idempotency key, bounded exact identity and cutoff.
3. `migrations/20261031_orotitan_runfactory_p0e_shadow_initial_run.sql`:
   SHADOW-only, service-role-only atomic transaction that locks issuer/cutoff,
   validates exact issuer/security/active dossier, verifies all pin statuses,
   reuses official `create_orotitan_run` and `bind_orotitan_run_identity`
   RPCs, rejects collisions, ensures two events and no stage or publication,
   writes auditable RLS-protected `orotitan_p0e_run_admission_receipts_v1`.
4. `app/admin/research-factory/run-preflight/commit/route.ts`:
   exact hosted-SHADOW env + admin session + same-origin + explicit V3.1 GO,
   fresh Registry reads, all 15 remote byte checks, one transactional RPC,
   postcommit independent run/event/stage/snapshot/receipt readback.
5. P0-B operator interface: prominent explicit command and receipt.
6. `tests/runfactory-p0e-shadow-plan-v1.test.ts` and
   `tests/postgres/run-orotitan-p0e-shadow-run-tests.sh`
   plus GitHub Actions workflow for Node/PG17 tests.
7. `scripts/orotitan-p0e-verify-contract-bytes.ts`: independent,
   read-only original-byte verification.

## Acceptance and evidence

A test is PASS only when its tool exit code is 0 and expected receipts,
readbacks and counts match. Never infer production results from a SHADOW test.
The full original byte gate is separate from the SQL contract-set hash check.

The first persisted run must be:
- One row under issuer/cutoff and exact V3.1 SHA;
- Security and dossier fully bound; state_version = 2;
- Event types RUN_CREATED and RUN_IDENTITY_BOUND once each;
- run_status CREATED; current_stage NULL;
- One P0-E admission receipt; 15 original bytes checks recorded;
- 0 stages, 0 published result, 0 snapshots in L’Oréal dossier;
- No production change.

There is no Research stage creation, no research analysis, no SOURCE VAULT
ingestion, no Deep Dive, no publication. Each requires its own GO and checks.

## Next step

P1-A: immutable source acquisition and source-lineage storage.
A subsequent `GO RESEARCH` must be separate from this P0-E creation.
