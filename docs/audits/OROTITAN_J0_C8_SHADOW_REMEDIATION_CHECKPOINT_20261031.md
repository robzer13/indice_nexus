# OroTitan — J0/C8 SHADOW REMEDIATION CHECKPOINT — 2026-10-31

**Status: PARTIAL / FAIL-CLOSED. C8 NOT ACCEPTED.**

This checkpoint documents actual execution and distinguishes installed verified
components from experimental changes **not** deployed or qualified. It does
not authorize or perform milestone acceptance.

## Authorized scope

User GO: design, implement and test an explicit, immutable, idempotent,
human-reviewed acceptance gate in SHADOW. **No acceptance of J0, no J1, no
Research, no publication, no production changes.**

- SHADOW Supabase: `awgsurdyvsyolcgpnygh`
- Excluded production Supabase: `cugpgtzygqqlxetyeven`
- L’Oréal run: `204390a1-5cba-4b9c-9821-ca0c7df43eca`
- Pinned qualified application source commit:
  `1c43b96dbd97d58c55fe0c0dd7dfa98fad28f2d2`

## Installed on SHADOW (verified)

1. `public.orotitan_j0_c8_acceptance_control_v1` has one global row,
   **`enabled=false`**, authority reference **NULL**.
2. `public.orotitan_j0_c8_reviews_v1` is an append-only review register.
   **0 reviews**.
3. `public.orotitan_j0_c8_decisions_v1` is an append-only approval register.
   **0 decisions**.
4. C8 SQL checks cover eight criterion labels, require prior evidence for
   C1–C7, and treat C8 as a **separate human decision**, not a fake
   self-certified eighth proof.
5. C8 review seals seven explicit evidence IDs, criteria SHA-256, audit
   reference/SHA-256 and journal watermark. Replays use stable
   idempotency keys. Server-side preparation checks the original audit
   Markdown bytes against SHA-256 from a fixed immutable GitHub commit.
6. Approval RPC requires explicit typed confirmation, matching review
   SHA, unchanged evidence/gates/journal watermark, zero unresolved
   issues, and a distinct authorization control. It is presently
   **locked**, no user GO has activated it.
7. Event serialization trigger
   `orotitan_j0_c8_serialize_event_insert` is enabled in SHADOW;
   its advisory lock key matches the C8 prepare/approval lock.
8. A real SHADOW `BEGIN ... ROLLBACK` journal insertion under
   `service_role` succeeded after serialization was installed;
   independent readback verified **0 test events persisted**.

## Actual test results

- Isolated PostgreSQL integration test of the original C8 engine: **PASS**.
  Review, seven proofs, replay, conflict, stale review, invalid GO,
  rejection while disabled, one fictional **J1** approval after a local
  database-owner test-only unlock, immutable receipt, and 0 J0 approvals.
  The disposable database was removed after testing. This is **not**
  a real SHADOW acceptance.
- Node application suite on pinned source: **745 passed, 0 failed**.
  TypeScript typecheck and ESLint: **PASS**.
- SHADOW migration `orotitan_j0_c8_human_acceptance_shadow_20261031`:
  applied, with successful preflight `BEGIN ... ROLLBACK`.
- SHADOW migration `orotitan_j0_c8_gate_event_serialization_shadow_20261031`:
  applied, with successful trigger rollback smoke.
- Direct SQL access control: `anon` and `authenticated` cannot read or
  insert decisions; `service_role` can read but **cannot directly
  INSERT** reviews/decisions or UPDATE the acceptance control.

## Remaining blocker — important

The original global kill-switch is deliberately disabled. Its schema is
not a sufficient general-purpose **per-milestone GO authorization** by
itself. An additional per-milestone authorization design has been
prototyped in source, but its migration/integration tests are **NOT
qualified and NOT applied in SHADOW**. Do not deploy that prototype, do
not unlock the global switch, and do not infer that J1–J8 will be
safely authorized by a GO for J0.

The existing C8 incident
`issue:j0-c8-human-acceptance-transition-not-implemented-20261031`
must remain **OPEN** pending the scoped authorization hardening and a
real authenticated Preview preflight/readback.

The admin reviewer/approver label is declared by the operator using the
existing shared admin session; it is not a cryptographically
verified individual identity. This limitation also must be disclosed
before any eventual production-grade governance release.

## Readback after installation

- `global_approval_enabled=false`, `activation_authority_ref=NULL`
- `reviews=0`, `decisions=0`, `accepted_milestones=0`
- `serialization_trigger=1`
- `L’Oréal stages=0, artifacts=0, dossier snapshots=0`
- `C8 blocker=OPEN`, J0 initial status `EN_COURS`, calculated
  operational status `BLOQUE`

## Recommendation — NO-GO to accept J0

1. Qualify and deploy **an additive per-milestone authorization gate**
   that cannot enable J1–J8 when only J0 receives a GO. It must never
   weaken the existing global-denial behavior.
2. Retest a fictional accepted milestone in disposable PostgreSQL with
   the final per-milestone authorization, role-based denial, and
   concurrent blocker/event serialization.
3. Validate hosted authenticated C8 preflight UI and immutable receipt.
4. Independently re-read SHADOW and only then close the C8 mechanism
   blocker. Do not accept J0 as part of this act.
5. A **different, explicit GO** will be required later to activate and
   execute an actual human acceptance of J0.

No J0 acceptance, stage start, snapshot, publication or production
deployment has been performed in this checkpoint.
