# D29b — Besi Research private intake adapter (review-only candidate)

**Branch:** `review/orotitan-d29b-private-research-intake-adapter-20261023`
**Base:** PR #403, not main. **No deployment, SQL migration, RLS/GRANT, storage bucket or public MCP route is created.**

## Security review of PR #403

PR #403 passes CI and correctly rejects malformed draft bytes and stale caller-supplied
versions, but its injected `ResearchIntakePort` cannot independently guarantee:

- Server-validated grant identity, issuer/run, revocation and short TTL.
- **Live** Registry run+stage CAS and atomic receipt-plus-audit at commit time.
- Private immutable byte storage with independent reread integrity verification.
- Concurrent duplicate reconciliation, ambiguous DB commit handling and orphan cleanup.
- Delegated OAuth storage access: generic OIDC profile is not write authorization.
- Independent D2 verification of **actual root-source bytes** and admission.

None of these gaps is resolved by merely turning on `mcpWriteEnabled`.
The code added here addresses the byte and adapter boundaries while preserving
the privilege separation.

## Implementation delivered

`research-private-quarantine-adapter-v1.ts`

1. Content-addressed private path rooted at run/RESEARCH/SHA-256; random UUID
   per attempt eliminates shared-path overwrite races.
2. Actual injected Supabase Storage `upload(...,{upsert:false})` and exact
   `download(path)` calls. No `list()`, signed URLs or public reads.
3. Checks in-memory SHA before upload and independently recomputes SHA after
   downloading the uploaded bytes. A corrupted or missing file cannot produce
   a receipt.
4. Calls preflight before allocating storage, then private serializable
   receipt+append-only-audit transaction adapter after integrity verification.
5. If the DB commit fails ambiguously, **never delete the upload on the request
   path**: a receipt could have committed. An independently authorized
   background reconciliation job must prove non-reference after a safety TTL.
6. Keeps outputs strictly `RECEIVED_NOT_VERIFIED`, `QUARANTINE_ONLY`,
   `admitted=false`, `mayAdvance=false`, `mayPublish=false`.
7. D2 independent reader only receives narrow metadata from the DB transaction.

`research-private-pg-receipt-adapter-v1.ts`

A private pool client executes only parameterized calls to proposed private
`preflight_v1`, `commit_receipt_v1` and `verifier_metadata_v1` routines,
in `REPEATABLE READ READ ONLY` or `SERIALIZABLE` transactions after a
`SET LOCAL ROLE orotitan_research_intake_writer` gate. Any private routine
failure rolls back and releases the client; no automatic serialization retry.

**These three SQL functions, tables and the role are NOT deployed or implemented
in the database by this PR.** They are *interfaces*, not a claim of working
database persistence. Their implementation needs a separately reviewed
private-schema SQL design, privilege/RLS review, real tenant/issuer authorization,
replay/revocation enforcement and staging tests.

### Supabase controls requiring independent review

Supabase Storage docs specify `upsert=false` to avoid replacement; new
private bucket should restrict reads and writes to the private service identity,
not `anon` / `authenticated`. Never grant bucket-wide listing to the public
OAuth/MCP path. Never manipulate `storage.objects` directly to delete blob
records: Storage metadata is not the underlying object store.

Docs verified 2026-10-23:
- https://supabase.com/docs/guides/storage/uploads/standard-uploads
- https://supabase.com/docs/guides/storage/security/access-control
- https://supabase.com/docs/guides/storage/schema/design

## Test boundary and matrix

In-memory fake private blob and serializable receipt transaction test the
contract and failures: exact Besi run, duplicate with only one receipt/audit,
wrong OAuth client, stale stage, revoked grant before AND after upload,
reread hash corruption, ambiguous transaction result, forged stored object
path, verifier isolation, no overwrite option. A separate PG query mock
checks transaction mode, scoped query params, rollback and release.

**No live OAuth, RLS, KMS, Supabase Storage, PostgreSQL or D2 end-to-end test
has run.** CI success is not an operational security approval.

## Requirements before deployment (not authorized here)

- Security review of #403 and this stacked PR, including actual authorization
  source and test of revoked/stale grants (OAuth sub/client/resource).
- Review-only private SQL schema and functions using `SECURITY INVOKER` and
  auditable transactional CAS, with least-privilege role and no public access.
  Run/admission states must remain unchanged by the receipt writer.
- Independent privileges and authz for the private writer and separate D2
  verifier. No use of public JWT `user_metadata` for authorization.
- Real PostgreSQL 17 negative tests of wrong run, concurrent stage mutation,
  serialization races, actor/client mismatch, revoked grant, duplicate object
  and missing atomic audit event.
- Real private Storage test with stored-byte hash verification and orphan
  reconciliation. Orphans may only be garbage-collected after proving that
  no committed receipt references the blob.
- Independent Research evidence verifier and immutable Ledger/Manifest review,
  with separate explicit authority to admit evidence or change Registry state.

Do not merge into main, expose a new OAuth write tool, enable a flag,
provision a billable service, or advance Besi RESEARCH as part of this PR.
The six Besi gaps remain OPEN_UNVERIFIED; the last Registry observation was
run state 6, stage state 4, gate READY_FOR_DEEP_DIVE=NO.
