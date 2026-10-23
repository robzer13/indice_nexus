# D29b Research private PostgreSQL receipt gates — SECURITY REVIEW ONLY

**Stack:** PR #403 (typed intake) -> PR #404 (private Storage/PG adapters)
-> this PR (SQL proposal + isolated PostgreSQL 17 test fixture).
**Status:** draft proposal. **No SQL applied to Supabase production**.
No private bucket, OAuth write scope, Registry mutation or deployment.

## Contract boundaries

The SQL proposal `D29B_RESEARCH_PRIVATE_RECEIPT_SQL_REVIEW_ONLY.sql`
introduces three functions in a deliberately non-exposed private schema:

- `preflight_v1` — independently checks a server-owned, short-lived,
  nonrevoked submission grant; exact OAuth subject/client/resource, issuer,
  Besi run UUID, `RESEARCH` stage, live run/stage CAS, date cutoff and
  `contract_set_sha256`. `BLOCKED_INSUFFICIENT_INPUT` is **not** excluded:
  the point is to admit draft uploads while Research remains blocked.
- `commit_receipt_v1` — requires a SERIALIZABLE transaction and private
  writer role, repeats the grant/live-CAS checks after its run-scoped private
  advisory lock, validates metadata including cutoff and object path,
  inserts a content-hash-unique, append-only `RECEIVED_NOT_VERIFIED` receipt,
  and enforces a same-transaction append-only audit trigger.
  Duplicate SHA+versions returns the original receipt; mismatched filename,
  issuer, client, subject or size fails closed.
- `verifier_metadata_v1` — requires a separate private D2 reader role
  and a live, short-lived, run/issuer/version-scoped reviewer grant, and
  returns only quarantine metadata; no source bytes, stage authority or
  bucket-wide list.

The TypeScript adapter is revised to require **distinct writer and verifier
pools** and issues `SET LOCAL ROLE orotitan_research_intake_writer` vs
`SET LOCAL ROLE orotitan_research_intake_reader`; neither role is a member
of `anon` or `authenticated`, and this PR grants membership to neither.

Tables and functions are located in `orotitan_research_intake_private`.
No publication- or admission-capable schema is touched. New narrow SELECT
policies allow the *private roles* to read the Besi run/stage. They are
**proposed only**, not applied to any live Registry.

## Live release blockers / known limitations

1. **No grant issuer, authenticated writer service or verifier service is
   deployed.** Grants in the test are inserted solely by PostgreSQL fixture
   superuser. The real issuer must be privately authenticated, approved,
   audited and revocable. Generic OIDC claims or `user_metadata` do not
   confer WRITE permission.
2. **No guarantee against every Registry-update race yet.** SQL uses a
   SERIALIZABLE transaction and private advisory lock, but the existing
   Registry writer may not take the same lock. A concurrent stage change
   could serialize after a draft receipt. The independent verifier MUST
   re-read current CAS before accepting anything. Production approval
   requires real concurrent two-transaction tests against the full Registry
   writer and agreed lock protocol or equivalent stronger enforcement.
3. **Blob/DB are not a distributed atomic transaction.** The #404 Storage
   orchestrator preserves unreferenced uploaded bytes if a commit fails
   ambiguously; cleanup requires an independently authorized reconciliation
   service proving an object is unreferenced after a safety TTL. Duplicate
   retries also produce orphan candidates, not an authorization to delete.
4. **A trusted private writer remains privileged** for inserting receipt
   rows. SQL includes an INSERT audit trigger but server-side grant/CAS checks
   can be bypassed by a compromised private role directly inserting into
   the table. Before production, redesign direct table DML away from the
   runtime role (e.g. a narrowly owned vetted stored procedure), or secure
   equivalent, with independent privilege and tamper tests. A production
   private role cannot be considered least-privileged until then.
5. The SQL is not a migration and deliberately does not construct a private
   Supabase Storage bucket, a token exchange, or any endpoint in the MCP
   route. Supabase RLS/policies, role memberships and grants need a full
   independent production-specific review before deployment.
6. No authentic original Research source bytes, ESEF archives or Research
   Evidence/Conflict/Calculation Ledgers are read or admitted. The Besi
   6-blocker worklist remains `OPEN_UNVERIFIED`, run/stage `6/4` at last
   observation; Deep Dive handoff is still NO.

## Verification scope

`bash tests/postgres/run-d29b-research-private-sql-tests.sh` is integrated
into `npm run test:postgres`. It refuses non-loopback hosts, non-fixture
user/port and PostgreSQL other than 17; allocates a disposable database,
creates synthetic Registry rows, applies review-only SQL there, and tests
authorized and denied preflight, cross-run isolation, revoked grants, wrong
client, stale stage, source cutoff, duplicate idempotency, exactly one audit
record, no Registry mutation and an independently scoped verifier.
The script drops the disposable DB and roles after testing.

This is *test fixture* verification, **not a Supabase production test**.
Passing CI and review-only SQL are insufficient to enable a write scope,
merge into main, advance Besi, attest an immutable artifact or publish.
