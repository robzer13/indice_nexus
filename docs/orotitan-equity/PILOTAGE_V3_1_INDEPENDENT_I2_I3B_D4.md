# OroTitan Pilotage V3.1 — D4 independent candidate recomputation

Status: REVIEW ONLY; code proposal in **robzer13/indice_nexus**, draft PR #383. No frozen-contract rewrite, database migration, production deployment, publication, or merge.

## Deliverables

- lib/orotitan-equity/v3/pilotage-integration-recompute-v3-1.ts
- tests/equity-v3-pilotage-integration-recompute.test.ts
- D2 trusted-loader wiring to validate only the exact currently registered FINAL Integration candidate bytes after manifest, size, SHA-256, Git blob identity and output linkage have passed.

## What D4 can prove

1. Authenticated candidate identity (issuer, security, dossier) and eligible analysis mode, received only via D2's server-owned verified-content path.
2. Independently evaluate the same frozen V1 I2 engine through the composed V2 validator, which recomputes the seven quality inputs, OQS/OVS, investment score and terminal status against the candidate's persisted deterministic outputs. The recomputation rejects mismatches and unsupported semantic states; it does not accept a textual PASS flag.
3. Validate the entire V3.1 additive schema using JSON Schema Draft 2020-12 and verify exact cutoff/valuation-date, share-count date, reference-price date, denominator representation, positive exact count or ordered finite bounds, and applicable market-cap date.
4. Where available in D2-verified and D3-checked VALUATION_ARTIFACT bytes, cross-check valuation date, share-count representation, FCFF versus equity DCF bridge rules and independent valuation-date-alignment resolver.
5. Reject malformed/missing output, divergent upstream records, wrong reference-price date, unknown denominator for a finalized analytical candidate, hidden illegal scalarization in the explicit count representation, a non-analytical DISCOVER candidate and missing required overlay fields.

## What D4 cannot certify

* I2 is recomputed **from the candidate's own canonical analytical inputs**. This is a valid deterministic consistency check, but does NOT prove that those inputs exactly match all upstream certified full-precision Deep Dive data. That proof requires a frozen, machine-readable full-precision projection and exact content linkage. D4 therefore reports \`i2CandidateRecomputed=true\` but **always** \`i2Pass=false\` for full end-to-end admission.
* The V3.1 denominator flags are checked for structure and date, but their underlying economic provenance (complete movement classes, hard-bound correlation, absence of stale transport) is not independently established by D4.
* The historical snapshot transition, current dossier pointer and transactional compare-and-swap are not verified by this pure read-only module. Consequently \`historyCasValidated=false\`, \`i3bPass=false\` and \`canonicalCandidateAdmitted=false\`.
* D4 does not infer admission from an I2_RECONCILIATION_REPORT or I3B_ADMISSION_REPORT carrying a self-declared PASS.
* \`publicationAuthorized=false\` always. D4 performs no mutation, makes no RPC call, changes no canonical snapshot pointer, and exports no public HTTP endpoint.
* The existing D1 Pilotage guard continues to withhold \`I2_PASS\`, \`I3B_PASS\` and \`CANONICAL_CANDIDATE_ADMITTED\` until an independent end-to-end attestation boundary can actually prove them. Successful D4 candidate checks alone cannot yield GO PUBLISH.

## Required follow-up gate

Before production activation, add verifiable upstream full-precision scoring lineage and an explicit read-only history transition snapshot with expected pointer. The eventual writer must enforce the actual compare-and-swap transaction using the pinned SQL and a distinct explicit GO PUBLISH permission; a prior read cannot replace database CAS.

Test isolated fixtures and GitHub Actions CI are not evidence of a successfully validated live V3.1 run.
