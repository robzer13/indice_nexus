# D35 — Internal cross-PR authorization/security review (D30–D34, PR #410–#414)

**Date:** 2026-10-23. **Disposition:** review-only / stop-ship for real MCP writes.
**Review scope:** typed mobile intents (D30), signed OAuth identity / HMAC capabilities
(D31), private D32 SQL approval/submitter boundary, D33 actual ES256-to-PG17
integration, D34 encrypted preview/vault roles.

This is an **internal engineering review**, NOT independent penetration testing,
a live Supabase audit, mobile acceptance testing or production authorization.
No main merge, deploy, Supabase migration, new paid branch, OAuth write scope,
company insert, Research run, Registry stage transition or Besi evidence admission.

## Findings and remediation

| ID | Severity | Finding | This PR / disposition |
| --- | --- | --- | --- |
| D35-01 | HIGH (preproduction) | D34 preview issuance accepted any flat string-only JSON payload with a matching computed fingerprint. A caller with an issuer role could submit arbitrary extra keys or a Research identity without checking the canonical I1 issuer/security relationship until the later D30 confirmation. | **Fixed in proposal:** require exact 4-key onboarding or 5-key Research schema, syntax and current I1 canonical Research crosswalk; reject already tracked company identities. Tests exercise valid/fake cases. No independent market provenance claim. |
| D35-02 | MEDIUM | D34 private preview read returned a confirmed preview until its two-minute expiry, although encrypted vault read correctly stopped after confirmation. | **Fixed in proposal:** `read_preview_v3` now requires `confirmed_request_id IS NULL`. PG17 E2E asserts NULL after successful confirmation. |
| D35-03 | HIGH / BLOCKING | Private SQL `create_preview_v3`, `read_preview_v3`, `approve_and_store_capability_v3` and `read_encrypted_capability_v3` take caller-supplied `subject/client/resource`. A compromised or misplaced trusted runtime role can impersonate other actors by knowing their preview ID and actor values; `NOLOGIN` and `SECURITY DEFINER` alone do not make those values authentic. | **Unresolved.** Trusted server verification, actor-bound session assertion, isolated credentials and an independently audited binding to verified OAuth sessions are release blockers. NEVER grant to public/browsers or enable MCP commands now. |
| D35-04 | HIGH / BLOCKING | D30 global `UNIQUE(kind,target_key)` is permanent while the only state is `REQUESTED_NOT_EXECUTED`. This prevents repeated Research requests for the same issuer and rejects a different actor/client, potentially exposing a cross-tenant denial-of-service oracle. | **Unresolved.** Design a tenant/run/contract-cutoff-scoped request lifecycle and active idempotency keys with expiry and replay rules. Do not weaken deduplication ad hoc. |
| D35-05 | HIGH / BLOCKING | Identity resolver is synthetic and checks mostly format. There is no independent ISO 6166 check digit, official security/listing ownership, corporate-action/ticker-reuse mapping or dual-listed security reconciliation. | **Unresolved.** Independent issuer/security provenance + operator ambiguity review required before creating any real company/Research request. |
| D35-06 | HIGH / BLOCKING | No hosted deployment of the D30–D34 schema, OAuth write authority, real mobile connector test or representative Supabase Storage/RLS validation. | **Unresolved.** Separate scope/cost/security/deploy approval and mobile capability acceptance before writes. |
| D35-07 | MEDIUM / BLOCKING | AES-GCM envelope is test-only: no audited KMS/key rotation, server-only persistent decryptor, access log hygiene, one-time token lifecycle or revocation and retention policy. Synthetic `psql -c` tests transport token text through local process args and may log it on failures; never copy that transport mechanism to production. | **Unresolved.** Build reviewed secret lifecycle without command-line token interpolation. |
| D35-08 | MEDIUM | Seven local roles and multiple SECURITY DEFINER endpoints have a broad owner RLS policy (`USING(true)`) and could be unintentionally granted to `anon`, `authenticated` or the wrong internal principal during migration. | **Additional regression guard:** new D35 PG17 ACL matrix checks all seven `NOLOGIN/NOINHERIT/NOBYPASSRLS` properties, function ownership and search path, direct table access refusal, correct EXECUTE separation, private RLS and no anon/auth grants. Actual hosted grants still need independent audit. |
| D35-09 | MEDIUM | The SQL currently uses selected per-issuer rows under an all-rows executor policy. A bug or compromised executor would have more read scope than a genuine per-user/per-run RLS policy. | **Unresolved.** Design tenant-bound RLS and test cross-tenant reads in real auth contexts prior to release. |

## Test coverage added

1. Real D34 PG17 confirmation closes both vault read AND preview read.
2. D34 issuer SQL rejects an extra `permission` payload field even with a
   freshly computed, valid SHA-256 preview fingerprint.
3. D34 issuer SQL rejects a well-shaped Research request when its
   `issuer_id/name/ticker/exchange/isin` do not match current I1 identity.
4. An independent D35 SQL matrix runs in the disposable D32/33/34 database.
   It checks owners, role grants, cross-role EXECUTE, no direct table access,
   public `anon/authenticated` denial and RLS enabled on all private tables.
5. The full previous D30–D34 unit, concurrency and ES256/OAuth integration
   suites remain required in GitHub Actions.

**Explicit note on idempotency:** successful SQL CAS under concurrency is
not a complete safe queue lifecycle. The D30 global uniqueness and permanent
`REQUESTED_NOT_EXECUTED` state prevent production readiness.

## Release decision

**NO-GO** for deploying this write-enabled MCP to ChatGPT mobile or hosted
Supabase. Passing isolated PostgreSQL/ES256/AES-GCM tests proves only the
specific synthetic paths checked. Independently review role/session binding,
multitenancy/lifecycle, market identity, encrypted-key management, recovery,
redaction, failure paths and actual connector support. The deployed MCP remains
read-only; Besi's six Research blockers and Deep Dive gate must not change.

**No additional Supabase subscription expense or billable branch approved.**
