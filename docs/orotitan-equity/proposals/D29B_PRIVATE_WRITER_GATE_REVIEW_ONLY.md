# D29b — Private receipt writer gate, review only (D29b/W1)

**Status:** PROPOSAL; no database DDL, no writer role, no production Vercel deployment.  
**Base:** validated D29b issuer + delegated reader `dc96dde4a5a478da9868258a015496da44cae23a`.

## Responsibility separation

- `mcp-d29b-d2-issuer-server.ts`: the independently authenticated administrator calls D2 immutable evidence verifier and receives an Ed25519-signed receipt from an injected KMS adapter. No DB write.
- `mcp-d29b-private-writer-boundary-v1.ts`: before any persistence, verifies the signed receipt with the trusted PUBLIC key, exact live OAuth subject/client/resource/run scope, exact Registry state versions, expiration, and distinct admin operator/approval IDs. **Requires** an injected, single-transaction `storeWithAudit` implementation. The standalone code has no Supabase client and does not grant any write permission.
- `mcp-d29b-delegated-reader-v1.ts`: uses only the caller's OAuth JWT and publishable key under RLS; checks state twice and receipt presence twice. No privileged signer or write access.
- The separate proposal `D29B_SIGNED_RECEIPT_RLS_SQL_REVIEW_ONLY.sql` creates the read-scoped receipt table and revocation predicate. It remains **unapplied**.

## Atomic writer contract — mandatory before real activation

A subsequent reviewed implementation must perform all steps within the *same* PostgreSQL transaction under a dedicated non-login, internal writer role (never the OAuth `authenticated` role or a public service endpoint):

1. Validate a previously approved operator and unique, revocable approval reference using server-controlled records.
2. Lock the selected run/stage as needed and verify exact run/stage versions, stage, lifecycle and approved pilot restrictions; fail on CAS drift or existing blockers.
3. Write at most one receipt for the exact subject/client/resource/run identity. Do not rely on a public `UPSERT` that could create ambiguous receipts.
4. Append an audit event with hash of the signed envelope and signer key ID, operator, approval reference, target version pair, timestamp and outcome. Transaction must roll back the receipt if audit insertion fails.
5. Support revocation via `revoked_at` and cleanup/rotation without granting OAuth writes. Reject stale/expired and previously used approval references.
6. Return only `STORED`, `CAS_FAILED`, `AUDIT_FAILED`, `NOT_AUTHORIZED` to the injected adapter. Do not expose sensitive row contents in errors.

No `service_role`, private key or artifact token is introduced into the OAuth code path by this proposal. A future issuer adapter must use a separate protected runtime. *Never* take operator authorization solely from a ChatGPT/OAuth supplied argument.

## Required tests and release blockers

- Unit tests: positive verified receipt; forged signature; stale CAS; mismatched scope; existing blockers; expired receipt; absent approval; denied DB transaction/failed audit. All negative outcomes must cause zero or rolled-back writes.
- PG17 local tests: RLS authorized caller only, other-run denied, `revoked_at` immediate denial (already in PR #391); add future transaction/audit failure rollback tests when the private writer is implemented.
- Protected integration: KMS signing, issuer audit, real reader with caller JWT, revocation, live CAS and fail-closed fallback.
- No signable Besi Research handoff until evidence gaps resolved under immutable contracts; no automatic Deep Dive transition or publication.
- **Separate explicit approvals** required for Supabase DDL, writer role, KMS/secrets provisioning and Vercel pilot activation.

This change is a security boundary and a test harness, **not a working storage integration**.
