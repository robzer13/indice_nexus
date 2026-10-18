# OroTitan Pilotage V3.1 — D6 provenance proposal and transactional writer admission audit

**Date:** 2026-10-18. **Status:** PROPOSAL / REVIEW ONLY, draft PR #383 in `robzer13/indice_nexus`. No live writes, migrations, frozen contract changes, deployments, or publication.

## 1. Versioned I2 provenance proposal

Machine-readable proposal: `docs/orotitan-equity/proposals/OROTITAN_I2_FULL_PRECISION_PROVENANCE_SCHEMA_D6_V0_1_PROPOSED.json` (`schema_version: D6_PROPOSED_0.1.0`). The proposal is NOT the frozen methodology or a new valid physical Registry artifact type. It describes the *body* of the already named `SCORING_INPUTS_FULL_PRECISION` output.

Contents: `run_id`, `data_cutoff`, `stage_revision`, pinned `contract_set_sha256`, own UUID/version, exact `research_evidence_ref`, complete `canonical_contract_input`, and an array of explicit per-leaf bindings. Each binding specifies the exact I2 input JSON pointer, source artifact UUID/version/SHA-256/type/stage, source JSON pointer, and the role of the evidence authority.

Role map proposed for review:

| I2 input family | Source authority type | Role |
|---|---|---|
| Dimensions, evidence grades | Deep Dive `ANALYTICAL_BLOCK_OUTPUTS` | `FUNDAMENTAL_JUDGMENT` |
| Certification status and score permission | Deep Dive `CERTIFICATION_ARTIFACT` | `CERTIFICATION_JUDGMENT` |
| MoS/reliability and primary/normalized expected returns | Deep Dive `VALUATION_ARTIFACT` | `VALUATION_JUDGMENT` |
| Ten terminal elite-gate predicates | Deep Dive `OROTITAN_TERMINAL_GATE_ARTIFACT` | `TERMINAL_GATE_JUDGMENT` |

Research Evidence Ledger is required separately. The Fundamentals source JSON must link back to its exact UUID/version/SHA. The D6 validator requires every canonical input leaf to be represented exactly once, rejects unknown/missing/duplicate bindings, checks each source identity against D2 byte-verified Registry/FINAL manifest outputs, and compares the source value with the unrounded candidate-derived I2 input. It recomputes both sides with the existing frozen I2 algorithm. Source pointers cannot access unsafe object keys. Source records must agree on run, cutoff and stage revision.

**Scope of proof:** This is a *proposed* unambiguous extraction and source-binding shape. It does NOT independently certify whether management judgments, source prices, evidence assessments, DCF projections or economic share-count bounds were analytically correct. A source that merely repeats a candidate input is insufficient to prove underlying economics. The frozen contract must separately decide authority of source record contents and allowed source version/paths. `methodologyFreezeApproved=false`, `analyticalCertificationProven=false`, `i2Pass=false`, `i3bPass=false`, `publicationAuthorized=false` in all D6 results.

## 2. Writer inventory: live Supabase, read-only inspection

Project: `cugpgtzygqqlxetyeven` / `orotitan-screener`, region `eu-west-3`, status `ACTIVE_HEALTHY` during inspection. Catalog queried read-only using `pg_proc`, `pg_get_functiondef`, `pg_constraint`, `information_schema`, migration history; no SQL mutation executed.

| Gate | Observed production state on 2026-10-18 | Admission |
|---|---|---|
| Validated canonical snapshot writer V1 | `persist_orotitan_research_snapshot` installed | V1 only |
| Validated canonical snapshot writer V2 | `persist_orotitan_research_snapshot_v2` installed, `SECURITY DEFINER`, `service_role` EXECUTE | V2 only |
| Writer V3.1 | No function with a matching V3/V3.1 writer identity located; full public function inspection showed no V3 writer | **NO GO** |
| Physical schema support | `research_snapshots` CHECK allows only `04_SCREENER_SCHEMA_V1`/`1.0.0` and V2/`2.0.0` | **NO GO** for a distinct V3.1 pair |
| Publication orchestration RPCs | `record_orotitan_publish_authorization` and `record_orotitan_publish_result` installed | Not snapshot-writer substitutes |
| Transactional protection in V2 | `SELECT ... FOR UPDATE` on dossier, null-safe expected pointer CAS, idempotent exact replay, insert+pointer update in one function | Proven for V2, **not transferable to V3.1** |
| Production snapshot history | 1 V1 and 34 V2 snapshots; no persisted distinct V3.1 schema version | No V3.1 physical proof |

Installed migration history includes V1/V2 canonical snapshot writers and V3.1 *methodology successor-run* repairs, but **no V3.1 canonical snapshot writer migration**. The method-successor RPC creates/repairs a run; it does not validate and atomically publish a V3.1 snapshot.

Independent reproducible catalog checks (all SELECT):

```sql
SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS args,
       p.prosecdef AS security_definer,
       has_function_privilege('service_role',p.oid,'EXECUTE') AS service_exec
FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
WHERE n.nspname='public' AND (p.proname ILIKE '%snapshot%'
  OR p.proname ILIKE '%writer%' OR p.proname ILIKE '%i3b%')
ORDER BY p.proname;

SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint WHERE conrelid='public.research_snapshots'::regclass
  AND contype='c' ORDER BY conname;
```

## 3. Admissibility gate for a future V3.1 writer

Do NOT reuse V2 `persist_orotitan_research_snapshot_v2` against a V3.1 candidate. Approval requires ALL of:

1. Separate reviewed, pinned V3.1 physical contract/schema pair and frozen input provenance rules; historical V1/V2 pairs remain immutable.
2. A V3.1 writer that checks the exact V1 analytical core, V2 product, V3.1 overlay, I2 independent recomputation and provenance, exact denominator representation / bounds / no scalarization, FCFF/FCFE bridge and valuation-date alignment from authoritative bytes.
3. Dossier/issuer/security and baseline identity, current candidate/manifest authority, and exact input-graph provenance; no self-attested PASS report.
4. Atomic row lock (`SELECT ... FOR UPDATE`), expected pointer CAS checked inside the SAME transaction, immutable snapshot insert and current-pointer switch, exact idempotency and conflicting-replay refusal, no historical rewrite.
5. Explicit `GO PUBLISH <COMPANY>` from the user, a distinct guarded publisher capability, idempotent authorization/event/result path, least-privilege `service_role` execute and no exposure to `anon` or `authenticated`.
6. Negative SQL tests of concurrent stale pointer, divergent repeated ID, wrong identity/contract, missing gate, stale manifest and replay; approved migration through a separate explicit change review.

**Decision at D6:** `WRITER_V31_NOT_INSTALLED`; `PHYSICAL_V31_SCHEMA_NOT_ADMITTED`; `SOURCE_PROVENANCE_NOT_FROZEN`. Hence `NO GO` for publication. D1–D6 remain on draft branch; the proposed provenance schema passes only advisory technical validation and must not be elevated to official authority.

## 4. No production effect

No other GitHub repo (including any immutable artifact storage repo) was consulted. Supabase operations were SELECT/catalog introspection only. No run, artifact, document, snapshot, pointer, SQL function, policy, migration or deployment was altered.
