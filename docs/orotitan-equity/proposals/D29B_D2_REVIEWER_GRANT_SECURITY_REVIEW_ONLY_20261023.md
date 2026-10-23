# D29b D2 reader scope hardening — independent review findings (review only)

**Stack:** draft PR #403 → #404 → #405 → #406 → #407 → #408 → this PR.
**No production code path, no Supabase deploy, no OAuth write grant and no new billable environment.**

## Audited defect (HIGH): role-only impersonation of the D2 verifier

The previous `verifier_metadata_v1(run,issuer,reviewerId,versions,sha)`
trusted a text `reviewerId` passed by any holder of the private *reader*
database role. That role also had direct `SELECT` on the verifier-grant and
receipt tables. A compromised authorized reader could discover another
reviewer's ID, pass that ID in the RPC and extract their run-scoped receipt
metadata without the originally authenticated reviewer's authorization.
`reviewerId` is an identifier, **not an authorization credential**.

This PR requires a separate opaque `grantId` as well as reviewer identity,
issuer/run/stage, dual Registry CAS, live expiry and revocation. The reader
runtime cannot directly SELECT `receipts` or `verifier_grants`. The query
runs only through a separately owned `SECURITY DEFINER` function whose owner
(`orotitan_research_intake_verifier_executor`) is a non-superuser,
`NOLOGIN`, `NOINHERIT`, `NOBYPASSRLS` role with restricted SELECT and
run-scoped RLS. The invoker has only EXECUTE.

Production **MUST** obtain opaque grant IDs from a privileged trusted server,
using cryptographically random, non-enumerable values of at least 128 bits.
IDs must never derive from names or user-editable JWT metadata, and the D2
server must independently bind OAuth/service actor identity and the grant
before calling the database. The test fixture grant IDs are intentionally
not real secrets.

## Tests added/updated

- Previous passing authorized verifier lookup now requires explicit grant ID.
- Correct reviewer ID + **incorrect grant ID** must return zero rows.
- Guessed reviewer ID + valid grant ID returns zero rows.
- Wrong run, stale Registry versions and revoked grants remain denied.
- The reader role has **no direct SELECT** on private receipt/grant tables.
- Both PostgreSQL 17 test suites (synthetic CAS and the true Registry
  `pause_orotitan_stage` RPC race) use the updated private function contract.
- The TypeScript `TrustedVerifierGrant` type now requires an opaque
  `grantId`; the private PG query passes it as a bound parameter.

## Cost-limited Storage follow-up

User chose **Option A** (avoid Supabase charges). The environment executing
this review lacks Docker, so a real Supabase Storage container cannot be
validated here. Official Supabase CLI local development requires Docker or
an alternative compatible runtime. The previous #408 exercise uses the pinned
`supabase-js` SDK against an HTTP loopback emulator, **not** genuine Storage
authorization/RLS.

Possible next separate review-only CI gate: launch an isolated local Supabase
CLI stack on a container-capable GitHub runner, with explicit synthetic seed,
a private bucket, unauthorized `anon/authenticated` access checks,
server-only writer grant checks, and independent verifier checks. Pin the CLI
and image versions, keep the stack loopback-only, audit resource consumption,
and never use Supabase API credentials or hosted project references. It must
not be presented as verified until the real Docker tests pass.

## NO-GO for production

An opaque reviewer grant helps prevent the discovered identifier-only
impersonation; it does **not** replace server-side actor authentication,
server-controlled grant issuance/revocation, private credentials, Storage RLS
or independent verification of actual source bytes. The SQL remains a
REVIEW-ONLY proposal. No merge to main, no cloud deploy, no billing plan
change, no Registry write, no Research FINAL and no Deep Dive admission.
