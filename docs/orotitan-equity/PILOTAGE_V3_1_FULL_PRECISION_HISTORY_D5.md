# OroTitan Pilotage V3.1 — D5 scoring lineage and historical precheck

Status: REVIEW ONLY. PR #383 in robzer13/indice_nexus remains draft. No frozen contract changes, migrations, deployment, production writes or merge.

## 1. Exact I2 input lineage

D5 resolves the current Valuation Lock's full_precision_inputs_artifact reference against the same Deep Dive FINAL manifest's SHA-256-verified SCORING_INPUTS_FULL_PRECISION artifact. It requires an exact UUID/version/SHA match, same run and cutoff, JSON content, and validates the proposed canonical_contract_input structure against the existing frozen I2 input contract. It then compares exact scoring inputs with those derived from the immutable Integration candidate and recomputes both using the frozen computeCanonicalSnapshot engine. No rounding tolerance or narrative report PASS is accepted.

**Critical limitation:** canonical_contract_input is a proposed D5 extraction envelope, NOT a frozen Deep Dive artifact-body schema. Identical inputs do not themselves independently certify Fundamentals/Valuation numerical provenance. The module explicitly returns authoritySchemaFrozen=false, upstreamCertificationVerified=false, i2EndToEndPass=false and publicationAuthorized=false. Missing data fails closed.

## 2. Historical state checks

The D2 port is extended with SELECT-only reads from research_dossiers and research_snapshots. D5 verifies current dossier/issuer/security identity, baseline snapshot ownership, and for a new candidate that the current pointer equals the run baseline and the requested snapshot ID is unused. For an exact retry, the existing candidate must be the current pointer and its identity, versions, metadata and canonical JSON payload must match. Dossier and snapshots are re-read after checking for changes.

**Critical limitation:** read/verify/re-read cannot replace an atomic SELECT FOR UPDATE plus compare-and-swap in the authoritative SQL writer. D5 always returns transactionCasValidated=false, historyTransitionPass=false and publicationAuthorized=false. No V3.1 writer deployment or API route is assumed.

## 3. Files and acceptance

- lib/orotitan-equity/v3/pilotage-evidence-history-v3-1.ts
- lib/orotitan-equity/v3/pilotage-integration-recompute-v3-1.ts
- lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1.ts
- lib/orotitan-equity/v3/pilotage-trusted-loader-server.ts
- tests/equity-v3-pilotage-evidence-history.test.ts

CI must include lint, typecheck, TypeScript tests, PostgreSQL suites and build. The PR is non-production and does not inspect the separate artifact-hosting repository.

## Remaining activation gates

A formally approved and frozen full-precision/provenance body schema, a proven V3.1 validated snapshot writer and transactional history/CAS enforcement, plus separate explicit GO PUBLISH authorization, are required before I2_PASS/I3B_PASS/READY_TO_PUBLISH can be admitted. D5 never grants those gates.
