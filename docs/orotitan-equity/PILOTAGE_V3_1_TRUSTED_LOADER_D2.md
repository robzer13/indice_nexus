# OroTitan Pilotage V3.1 — D2 trusted read/verification loader

Status: REVIEW ONLY. Proposed in draft PR #383 in repository robzer13/indice_nexus, not deployed.

## Implemented components

- lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1.ts: injected server-owned Registry read interface, fail-closed schema and bytes verifier, stable two-pass state check, construction of a verified D1 input.
- lib/orotitan-equity/v3/pilotage-trusted-loader-server.ts: server-only admin-session guard, pinned production Supabase URL, service-role SELECT-only client port and allow-listed GitHub immutable byte reader. Exports no HTTP route.
- tests/equity-v3-pilotage-trusted-loader.test.ts: real SHA-256 and Git blob proofs; schema/manifest, canonical output set, tampering, storage coordinate mismatch, CAS race and wrong environment regressions.

## Read-only execution and verified evidence

The trusted loader verifies the exact V3.1 immutable contract pack, run identity, all current Registry stages and run-scoped artifact registrations. For an active Stage Manifest it retrieves bytes from exact registered PRIVATE_GITHUB repository/path/commit, recomputes content SHA-256, Git blob SHA-1, byte size and immutable storage URI, then parses UTF-8 JSON. It verifies the generic Stage Manifest V1 schema and exact V3.1 contract/identity binding, stage revision, current Registry lifecycle and handoff gate, and the full output-artifact set and all its bytes. It validates stage-to-stage parent manifest references when persisted, and re-reads the Registry rows and issuer identity to prevent returning a state already changed during verification.

The Stage Manifest V1 schema predates two legitimate RAW_CANONICAL_V1 locator metadata fields in the pinned V3.1 pack. For schema evaluation ONLY, these exact known fields are projected away AFTER comparison to the original complete pinned pack. No registered source or hash is rewritten.

D2 derives only:
- READY_FOR_DEEP_DIVE, READY_FOR_INTEGRATION or READY_TO_PUBLISH when the corresponding FINAL manifest is fully byte-verified and agrees with persisted Registry state;
- FINAL_OUTPUTS_VERIFIED only if every required canonical output is registered, linked, authoritative and its exact bytes verify.

D2 does NOT infer READY_FOR_VALUATION, READY_FOR_CERTIFICATION, DENOMINATOR_ADMISSION_VALID, VALUATION_DATE_ALIGNMENT_VALID, I2_PASS, I3B_PASS or CANONICAL_CANDIDATE_ADMITTED from the mere existence of documents. Dedicated typed content authorities and validators are required. Until present, D1 fails closed for those transitions; it cannot grant publication.

## Required server configuration

- NEXT_PUBLIC_SUPABASE_URL must be HTTPS on the exact production project_ref host specified by the active bootstrap.
- SUPABASE_SERVICE_ROLE_KEY, ADMIN_PASSWORD and the existing admin session secrets are managed by the current app security boundary.
- OROTITAN_PRIVATE_GITHUB_TOKEN must have read access to the immutable object store.
- OROTITAN_PRIVATE_GITHUB_REPOSITORIES is a **server-side allow-list** of the repositories permitted for persisted artifact reading; no implicit default.
- The trusted calling server must supply an independently observed RuntimeEnvironmentObservation verifying project name, region and health. The pinned hostname alone is NOT a substitute for independent environment confirmation.

No GitHub repository other than robzer13/indice_nexus was inspected or edited in D2 implementation. The application byte reader references its already persisted artifact storage coordinates only when invoked at runtime. No runtime invocation or data retrieval occurred during this code work.

## Known limitations and next gates

1. No HTTP route or public endpoint has been created. The server-only function requires an admin session. It should not be exposed as a generic caller-supplied decision API without its own rate limits, authorization and project-health observation.
2. Raw bytes and source metadata are verified. A full cryptographic validation of the entire upstream evidence graph beyond exact parent manifest pointers is NOT demonstrated by this loader.
3. Read-verify-reread is not a database snapshot isolation transaction. Any future writes must verify current versions through controlled Registry CAS and idempotency; D2 never writes.
4. Legacy or special manifests without explicitly supported immutable storage and exact V3.1 pins fail closed. Supabase Storage is intentionally not accepted here, because immutable content provenance would need an equivalent proof.
5. No certification of a real production run or artifact retrieval was attempted. Tests use deterministic local fixtures and GitHub Actions CI.
6. Run-state display-name normalization and multi-listing identity require additional real-run smoke checks before an HTTP integration. Handoffs retain exact RUN_ID and contract identities.
7. No contract freeze patch, no migration, no production deployment, no merge, no VNext cherry-pick and no publication.

Review D2 as a guarded foundation, not a claim of end-to-end production readiness.
