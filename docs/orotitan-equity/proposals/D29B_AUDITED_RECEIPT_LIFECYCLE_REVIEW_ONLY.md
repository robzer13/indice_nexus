# D29b — Audited receipt issuance lifecycle (DRAFT, NOT DEPLOYED)

**Base:** PR #391 `dc96dde4a5a478da9868258a015496da44cae23a`.  
**Scope:** synthetic test coverage and private architecture proposal only.  
**NO production changes:** no Supabase migration, GRANT, new role, KMS key, app deployment, Registry mutation, or Besi handoff.

## Components added

1. `mcp-d29b-receipt-lifecycle-v1.ts`: pure orchestration for a **previously issued D2** Ed25519 attestation. It rejects scope mismatch, altered signed expiry metadata, invalid signer, Registry CAS drift, atomic-store refusal, stale/missing readback and replay. A mismatch attempts a privileged revoke. Revocation is a separate admin-only action. This module **never writes directly** to a database and is not an OAuth action.
2. `mcp-d29b-d2-issuer-server.ts`: existing admin-session-gated privileged D2 issuer now additionally verifies the signature returned by a remote signing adapter against a **separately configured pinned Ed25519 SPKI public key** and exact signer key ID. No private key available to OAuth, client-side code or the public repo.
3. `D29B_PRIVILEGED_RECEIPT_WRITER_SQL_REVIEW_ONLY.sql`: **review-only** PostgreSQL 17 proposal for an isolated `NOLOGIN` role and schema, SQL `SECURITY INVOKER` routines requiring SERIALIZABLE transactions and verifying run/stage CAS, no Registry UPDATE, dedicated Besi-only RLS and an append-only audit table containing **only signed payload hashes**, not prompts. Authenticated/anon have **no access** to the private writer schema.
4. `run-mcp-d29b-audited-writer-tests.sh`: isolated disposable PostgreSQL 17 test; includes positive issuance, wrong-CAS denial, audit append and revoked receipt disappearing from OAuth RLS.

## Invariants

- No admin or signer path is imported by the OAuth route. The D29b feature flag remains explicitly `false` by default.
- Protected D2 checks exact immutable bytes, contract pins, manifest/lineage and CAS *before* this lifecycle may persist a receipt.
- Database writer is not a public callable interface. A separate privileged and authenticated service must be explicitly provisioned; the private schema is not exposed by PostgREST.
- Signature expiry and persisted expiry must agree. Short-lived receipts never authorize publication or stage transitions.
- Revocation hides the receipt under OAuth RLS; read-before/read-after checks are **not a serializable proof of future state**, so no guarantee against subsequent state changes.
- The example SQL role is **NOLOGIN** and deliberately not granted to a login, OAuth role, or the existing Screener application. Production provisioning remains a separate operation.

## Important limitations

This is still **not an operational release**. The following are missing and MUST precede any live enablement:

- Private, authenticated signing service and KMS/HSM-backed Ed25519 key whose public key is independently pinned; vendor/region/key policy, rotation and outage handling are not yet selected.
- Privileged writer DB connection and production implementation of the `D29bAdminWritePort` contract, using an actual SERIALIZABLE transaction and audit/revocation guarantees. No connection credential is created here.
- Production RLS/GRANT migration and rollback review, live negative access tests, separate permission to alter Supabase.
- End-to-end staging test of a **valid D2 evidence set**, including real D2 issuance and a supervised preview; test fixtures only prove refusal/acceptance mechanics, not real Besi evidence verification.
- Review of the private writer's serializability assumptions, platform runtime and external audit attribution; a transaction can always commit before a subsequent Registry change. Confirm the desired stronger consistency before relying on a prompt for execution.
- Separate explicit authorization before any deployment. No merge of unrelated PRs or bypass of Besi's current `RESEARCH` blockers.

## Test commands

- `npm run lint`
- `npm run typecheck`
- `npm test`
- `bash tests/postgres/run-mcp-d29b-audited-writer-tests.sh` on **loopback PG17** only

A successful CI does **not** establish production readiness and must never be represented as a deployed signing service.
