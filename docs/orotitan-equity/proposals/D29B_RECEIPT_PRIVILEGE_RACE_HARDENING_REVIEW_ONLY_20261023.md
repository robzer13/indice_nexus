# D29b — EXECUTE-only receipt writer and Registry-race hardening (review only)

**Stacked on** PR #405, itself stacked on #404 and #403, all pilot-only.
**No merge, migration, RLS grant, storage bucket, OAuth write scope, D2 admission or Vercel deployment.**

## Finding and remediation

The previous private SQL proposal granted `INSERT` on the quarantine
`receipts` table and `receipt_audit` table to
`orotitan_research_intake_writer`. An authenticated private caller in that
role could therefore bypass `commit_receipt_v1`, directly insert a forged
`RECEIVED_NOT_VERIFIED` receipt and trigger its audit without the grant
or live Registry checks. RLS scoped the run but not the complete authorization.

This PR removes **all direct receipt/audit table DML** and receipt-table
`SELECT` from that external writer. The role retains only the ability to
execute the guarded `preflight_v1` and `commit_receipt_v1` routines.
`commit_receipt_v1` is `SECURITY DEFINER`, owned by a separate private
`NOLOGIN NOINHERIT NOBYPASSRLS`
`orotitan_research_intake_receipt_executor`. This role alone has the
privileges to read Besi Registry/grant rows and insert/select receipts and
append audit events under explicit RLS policies. It cannot update Registry
state. No membership in this owner role is granted to the caller or a public
Supabase role. Function search path is explicit.

**Important trust boundary:** `SECURITY DEFINER` is only acceptable
because the definer is a narrowly privileged **non-superuser**, not
`postgres` or `service_role`; definer ownership, policies, EXECUTE
privileges and lack of public schema exposure must receive independent
review before production.

## Concurrency review

The private receipt path takes a private advisory lock and runs
`SERIALIZABLE`, but the existing Registry writer is not proven to share
this lock. The fixture now starts a Registry stage-version update in a
second PostgreSQL session, while a quarantine receipt transaction runs.
Because the order of serializable transactions can differ, the receipt may
legitimately be serialized *before* the stage change, or rejected as stale.
Both cases are safe ONLY if the receipt remains **unverified**, the audit is
atomic, and downstream D2 metadata lookup checks current CAS and refuses an
obsolete `runStateVersion/stageStateVersion`. The new test asserts those
properties after the other transaction commits.

**This is a true concurrent two-session PostgreSQL 17 test but a synthetic
Registry writer**, not a test of the production Registry state-transition RPC.
It does not prove linearizability across unrelated writers. The final release
still requires full Registry writer integration and a common lock protocol
or an independent admission-time CAS+immutable-artifact verifier.

## Tests

- External writer cannot INSERT or UPDATE `receipts` or INSERT audit rows.
- Private receipt executor has `NOLOGIN`, no `BYPASSRLS`, no superuser,
  no `CREATE` on private schema; `anon`/`authenticated` have no EXECUTE.
- Existing bespoke `preflight_v1` grant revocation and version controls.
- Normal submit and duplicate replay yield an immutable quarantine receipt
  with exactly one append-only audit event.
- Concurrent stage mutation and receipt attempt never grant admission.
- Post-race preflight and D2 reader deny stale stage version.

The new race test is invoked only within
`tests/postgres/run-d29b-research-private-sql-tests.sh` and refuses any
non-loopback host, non-fixture user/port or non-disposable database name.

## Outstanding before any live write

Independently provision and approve short-lived submission/verifier grant
issuers, secrets, private storage, role memberships, real RPC concurrency
checks, secure audit retention, orphan cleanup and byte-level D2 verification.
The user has not authorized production activation. Besi must remain
`RESEARCH/BLOCKED_INSUFFICIENT_INPUT` until independently admitted evidence
satisfies its frozen contracts. The six open blockers are unchanged.
