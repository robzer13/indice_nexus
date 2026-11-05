# OroTitan — J0/C8 HUMAN GATE — SHADOW v1

Date: 2026-10-31.
Source reconciliation: 2026-11-01, CODEX 01. The local qualification below
does not authorize a Preview deployment or any remote mutation.
Authority: **GO REMÉDIATION C8 SHADOW**; authorizes implementation,
installation and tests. **No authority to accept J0**, initiate J1/Research,
write an analytical artifact, publish or modify production.

## Functional invariant

A milestone's status now has three different meanings:

1. **Historical plan** `orotitan_j0_milestones_v1.initial_status`: immutable
   baseline. Never silently rewrite `EN_COURS` to `ACCEPTE`.
2. **Current operational readiness**: `orotitan_j0_c8_read_gate_v1`
   combines 7 prerequisite evidence classes, active blockers, and the
   current journal watermark. C8 is a separate human decision, not a
   self-referential prerequisite evidence.
3. **Formal human acceptance**: only a persisted
   `orotitan_j0_c8_decisions_v1` `ACCEPTED` row is authoritative.
   This row can be created only by an explicitly authorized RPC, after
   **two independently disabled gates** have been activated for a
   future, distinct user GO.

No individual run, stage, source vault, snapshot, publication pointer,
or existing methodical/contract pin is changed by this infrastructure.

## Files and ownership

- `migrations/20261031_orotitan_j0_c8_human_acceptance_shadow.sql`
  creates one locked config row, one immutable review table and one immutable
  decision table; read-only status RPC, proof validation RPC, review RPC,
  and acceptance RPC. All tables have RLS. No `anon` or `authenticated`
  access; `service_role` is SELECT-only on the new tables, not INSERT/UPDATE.
  Controlled RPCs use SECURITY DEFINER with explicit service_role EXECUTE
  grants and PUBLIC denial.
- `migrations/20261031_orotitan_j0_c8_gate_event_serialization_shadow.sql`
  adds a trigger that takes a transaction advisory lock for each journal
  event. Review and approval RPCs take the **same per-milestone advisory
  lock**, preventing a new incident from interleaving with decision
  eligibility/watermark validation.
- `migrations/20261031_orotitan_j0_c8_per_milestone_authority_shadow.sql`
  requires a separate exact-milestone lock and matching authority.
- `migrations/20261031211508_orotitan_j0_c8_shadow_independent_reviewer_generic_authority_v1_1.sql`
  reconciles the already-installed SHADOW migration with its original bytes.
  The global authority accepts `GO_ACCEPT_J[0-8]_...`; each milestone still
  requires its exact authority. Reviewer and approver declared labels must
  differ after `lower(btrim(...))`. Do not reapply this migration to SHADOW.
- `app/admin/research-factory/control-tower/c8-gate/route.ts`:
  authenticated, read-only preflight and cross-table receipt recheck.
- `.../c8-gate/prepare/route.ts`: authenticated + same-origin POST,
  exact 7 pinned proof references, SHA-256 audit bytes from an **immutable
  official GitHub commit**, immutable review and independent DB readback.
- `.../c8-gate/accept/route.ts`: separate human confirmation and GO reference,
  **disabled unless both** environment
  `OROTITAN_C8_ACCEPTANCE_GO_ENABLED=1` **and** DB
  `orotitan_j0_c8_acceptance_control_v1.enabled=true` (plus matching
  DB activation authority). Neither is activated by this work.
- `app/admin/research-factory/control-tower/j0-c8-gate-panel.tsx`:
  read-only preflight and review; the accept button appears only when
  the live gate is explicitly enabled and a review is prepared.

## Review preparation, deterministic and fail closed

The operator chooses a fixed vector of seven evidence event UUIDs
(C1 to C7), one per `operator_details.criterionIndex`, each of type
`EVIDENCE_RECORDED`, for the same milestone.
For J0, C1/C6/C7 must have a genuine
`VERIFIED_DIRECT_REGISTRY_READBACK` snapshot. The remaining C2–C5
may be `REPORTED` but need reviewer assessment; they are **never
automatically promoted to verified** merely because of an attachment.

The audit report reference must be a Markdown file on the official
`robzer13/indice_nexus` GitHub repository at a fixed 40-character commit.
The HTTP route retrieves original bytes from that exact immutable commit,
rejects redirects / timeouts / excessive size, and compares SHA-256
before calling the DB RPC. The SQL review receipt binds the supplied
SHA-256, seven event IDs, audit reference, milestone criteria hash, and
journal watermark. A second call with the same idempotency key returns
`REPLAYED`; changed payload is denied.

Review preparation requires **zero open blockers** and seven evidence
classes. As of installation on SHADOW, J0/C8 still has the pre-existing
blocker `issue:j0-c8-human-acceptance-transition-not-implemented-20261031`:
so real J0 review preparation is correctly **denied** until it is
independently resolved. No review has been prepared in SHADOW.

## Formal decision (FUTURE ONLY, no present GO)

A later explicit GO is required, separately from the C8 remediation GO.

The acceptance function requires all of:
- Both activation gates enabled; `service_role` cannot update the
  DB activation control directly.
- Matching `GO_ACCEPT_Jn_...` authority reference **already set**
  in the DB by a separate privileged, reviewed migration.
- An immutable previously prepared review ID plus exact review SHA-256.
- Explicit typed phrase `ACCEPTER Jx SHADOW` and a declared approver label,
  supplied from an authenticated admin session.
- Seven revalidated evidence UUIDs and unchanged milestone criteria hash.
- Current journal watermark strictly equal to the review's watermark,
  no new open blocker, no prior accepted decision.
- Unique decision key, unique accepted milestone, immutable decision
  and independent readback. Replays use the same deterministic client key.

`accepted` is a decision in the **new gate decision table** and never
an implicit permission to start J1, Research, or to publish.

## Security and limitations

- The operator label is **declared**. Current admin authentication does
  not provide cryptographically verifiable per-person identity; avoid
  describing this as a personal electronic signature. An organization
  seeking strong nonrepudiation should add named SSO principal attestation
  in a separate approved hardening phase.
- Database service role itself is trusted privileged infrastructure;
  it can call review/decision RPCs directly, but cannot turn on the DB
  approval lock. The HTTP route provides an additional runtime GO switch.
- `initial_status=EN_COURS` for J1 means old partial code exists,
  **not** that the J1 phase is authorized.
- Source/contract pins and publication authority are separate systems.
- No arbitrary audit-host URLs or GitHub domains are allowed via the
  protected HTTP review route.
- The risk of stale evidence is addressed by the journal watermark
  plus shared per-milestone transaction lock. A newer journal event
  invalidates a prepared review and requires a new human review.
- No production verification was performed, since production was
  excluded from the user GO.

## Local SOLO_OWNER candidate (2026-11-05, not activated)

The additive candidate `20261105_orotitan_j0_c8_solo_owner_attestation_shadow.sql`
adds a J0-only `SOLO_OWNER` mode without replacing `STANDARD_TWO_PERSON`.
It reuses Supabase Auth on the exact SHADOW project and requires, on both
`PREPARE_REVIEW` and acceptance, the existing admin session plus a server-side
`auth.getUser()` validation of an email-confirmed, non-anonymous user whose UUID
exactly matches the server allowlist. Neither email, UUID, provider nor owner
role is accepted from the request body. A publishable/legacy `anon` key is
accepted for Auth; `service_role` and secret keys are rejected.

The receipt seals `SUPABASE_AUTH`, the Auth UUID, `SOLO_OWNER`, and the immutable
amendment reference/SHA-256. No personal email is persisted. The amendment
ratification flag and database columns remain false/null by default, in
addition to the existing runtime, global and J0 locks. Consequently this local
candidate cannot prepare or accept a SOLO review until separate ratification
and activation work is explicitly authorized.

This attestation proves control of the configured Auth account and mailbox. It
is not civil-identity verification, an independent human review, qualified
electronic signature, or non-repudiation. Whether that assurance is sufficient
for SHADOW remains an explicit governance decision outside this candidate.

## Verification required for C8 remediation closeout

CODEX 01 qualification is reproducible with
`bash tests/postgres/run-orotitan-j0-c8-human-gate-tests.sh` against a disposable
local PostgreSQL instance (CI uses PostgreSQL 17). The script installs Registry
dependencies, then the four C8 migrations in the order listed above, ending in
v1.1. It replays all four while locks are closed and no decisions exist, compares
installed function/constraint/ACL/RLS/trigger fingerprints, then runs the local
matrix using the existing thirteen-pin V1 contract pack. The acceptance
simulation is **J0 only**; J1 evidence/review fixtures test denial and do not
start J1. No analytical stage or publication is created. The database is dropped
on exit. v1.1 reapplication with active locks/decisions must fail at preflight
without mutation. Never replay an obsolete predecessor alone after hardening.

The following historical hosted closeout steps are outside CODEX 01 and require
their own authorization; they are not prerequisites to run the local matrix.

1. Local disposable PostgreSQL: simulate evidence C1..C7, deny preparation
   without evidence, prepare, replay, conflict, deny acceptance by default.
2. In the disposable DB ONLY, activate control by the local database owner;
   deny wrong authority, wrong confirmation, stale review; create exactly one
   fictional **J0** decision with readback, replay, no stage, and no J1
   decision. Check all RLS/GRANTs and the normalized reviewer/approver guard.
3. Verify transaction serialization migration by a real SHADOW
   `BEGIN; ...; ROLLBACK` event insert; it must persist **zero** test events.
4. SHADOW post-migration readback:
   `enabled=false`, `authority_ref=NULL`, 0 reviews, 0 decisions,
   actual J0 blocker remains open, original L’Oréal run unchanged.
5. Compile Next.js, lint/typecheck, full test suite; Preview deployment READY.
6. Real authenticated admin GET C8 status on Preview (if necessary
   demonstrated by operator, as was C5); independently re-read DB.
7. Only after all six may C8's historical *missing mechanism*
   blocker be closed by an independent receipt. Closing C8 is **not**
   accepting J0. A separate later user GO is needed for acceptance.

## How to use (current mode)

1. Open authenticated SHADOW Preview `/admin/research-factory/control-tower`.
2. In **C8 · Acceptation humaine**, click
   **Vérifier les 8 critères et le verrou**.
3. Expected current result: J0 `BLOQUÉE` (one C8 blocker), seven
   prerequisite evidence categories covered, activation `FERMÉ`,
   0 reviews, 0 accepted decisions.
4. Keep the blocker open until the hosted panel is confirmed.
5. Do **not** activate any approval switch or enter a GO accepting J0
   under this remediation authorization.
