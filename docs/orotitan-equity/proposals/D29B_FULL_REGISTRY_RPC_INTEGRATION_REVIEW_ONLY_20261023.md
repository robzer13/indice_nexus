# D29b — Full Registry V3.1 RPC integration in disposable PostgreSQL 17

**Stack:** draft PRs #403 → #404 → #405 → #406 → this draft PR.

## Real Registry integration

Unlike the earlier **synthetic UPDATE** race in #406, this test loads
the actual canonical Registry migrations V1.1–V1.12 and the D11 V3.1
Research FINAL/Deep Dive successor proposal in an isolated PostgreSQL 17
database. It first executes the existing full D11 E2E, proving real
`create_orotitan_run`, Research CHECKPOINT, Research FINAL and
`start_orotitan_stage` RPCs with canonical pin-pack checks.

It then creates an **additional synthetic Besi-shaped run** locally
with the known Besi UUID, a real issuer identity from the disposable
fixture, a real frozen-contract hash, state counters `6/4`, a
REFERENTIALLY VALID Research CHECKPOINT artifact, and handoff `NO`.
It never reads or copies production Besi source files or evidence.

The new two-session test invokes the real
`public.pause_orotitan_stage(run_id,'RESEARCH',6,4,...)` function,
which updates **both** Registry state counters under Registry row locks
and writes the real Registry event. Simultaneously a serializable D29b
`commit_receipt_v1` attempts a draft-only insert.

## Required results

- Either commit-order outcome is tolerated: draft received first or
  rejected after the Registry CAS changes.
- The actual Registry stage becomes `PAUSED`, run and stage versions
  increase from `6/4` to `7/5`, and handoff remains `NO`.
- Quarantine receipts, if any, remain `RECEIVED_NOT_VERIFIED` with
  exact one-to-one append-only audit entries.
- A read for obsolete versions `6/4` by the separate D2 role returns
  no metadata.
- A real `start_orotitan_stage('DEEP_DIVE',...)` attempt rejects the
  CHECKPOINT/NO handoff, independent of quarantine receipts.
- An old submission grant cannot pass post-race preflight.

The test is wired into `npm run test:postgres` and runs entirely in a
disposable localhost PostgreSQL 17 instance under the restricted
`PGHOST/PGUSER/PGPORT` guard. No live Supabase project, storage
bucket, Vercel deployment, OAuth write path or Registry row is touched.

## Environment inventory (read-only, 2026-10-23)

Supabase lists an active `orotitan-screener` project and an inactive
`orotitan-vnext-shadow` project. No dedicated D29b preview branch was
listed at the time of review. Neither should be reused as the D29b
isolated test environment.

The next external integration requires a **dedicated development branch
or separate test project** selected with explicit cost/permissions
approval; the branch can incur hourly spend. It must receive synthetic
fixtures only and never production Besi data. A dedicated limited
database runtime principal, a private Storage bucket, verifiable hash
rereads, server-side grant issuers and authenticated D2 reviewer are
separate preconditions. Do not automatically apply the proposed SQL
migrations to an existing Supabase project.

## Explicit limitations

- A real local Registry RPC is stronger evidence than a synthetic
  single-row UPDATE, but this still does **not** exercise a hosted
  Supabase Storage transaction, real OAuth JWT signature validation,
  RLS under actual runtime credentials, clock-dependent grant
  revocation under latency, real KMS, or the signed D2 evidence
  admission pipeline.
- This local scenario is intentionally **not** the real Besi run; the
  reused UUID is synthetic and exists only inside a disposable DB.
- `SECURITY DEFINER` and RLS require independent review on any target
  production schema. No privilege escalation, admission or publication
  may be inferred from passing these checks.
- The run remains RESEARCH / BLOCKED_INSUFFICIENT_INPUT with six
  OPEN_UNVERIFIED blockers and Deep Dive handoff NO until a separate
  evidence-admission authority verifies original bytes.
