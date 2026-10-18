# Pilotage V3.1 — Read-only admission adapter (D1)

Status: PROPOSED, not deployed. Scope: robzer13/indice_nexus, based on main at b717b74dbdceacefb40de9e61dac3431693f309d.

## Authority

- Frozen Pilotage V3.1 incorporates V3.0 and V2.0 admission/handoff rules.
- Active main runtime is V3.0.3 / OroTitan V3.1.
- The V3.0.2 Contract Pin Pack binds 15 immutable authorities to SHA-256 3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2.
- Research V2.0 and Deep Dive/Integration V3.1 are not replaced by this adapter.
- Fundamentals, Valuation and Certification are distinct analytical phases of the same DEEP_DIVE Registry stage.

## Operation

The pure function planPilotageV31() receives a single authenticated and consistent Registry snapshot and returns an ADVISORY decision. It never executes a SQL RPC, creates a run, mutates a stage, changes the canonical pointer, computes analytical judgments or authorizes publication.

A trusted caller must separately and actually:

1. Authenticate access to the exact allowed Supabase production environment.
2. Load consistent Registry run/stage state with immutable contract pins, IDs, state versions and lineage.
3. Resolve exact archived artifact bytes and verify SHA-256, blob coordinates and availability.
4. Parse and validate authoritative manifest content and the gate flags attributed to it.
5. Supply the current predecessor stage revision consumed by the downstream stage.
6. Enforce transactional optimistic concurrency for any future operation outside this planner.

In particular, the boolean bytesVerified and gatesVerifiedAgainstBytes fields are input assertions, NOT cryptographic proof: never accept them from an untrusted client or expose this planner as a security boundary. Until a trusted server-side loader is implemented, keep this adapter disconnected from public mutation routes.

## Decision matrix

| Persisted state and proof | Advisory decision | Missing evidence |
| --- | --- | --- |
| Research NOT_STARTED/IN_PROGRESS | Continue Research | BLOCKED |
| Research FINAL verified; Deep Dive NOT_STARTED | Handoff Fundamentals | BLOCKED |
| Deep Dive Fundamentals CHECKPOINT + Valuation admission + lock | Handoff Valuation | BLOCKED |
| Deep Dive Valuation CHECKPOINT + new-run Valuation Lock | Handoff Certification | BLOCKED |
| Deep Dive FINAL verified; Integration NOT_STARTED | Handoff Integration | BLOCKED |
| Integration FINAL + exact candidate + I2 PASS + I3-B PASS | Prompt GO PUBLISH company only | BLOCKED |
| PUBLISHED | DONE | no new dispatch |
| BLOCKED/PAUSED, wrong contract, stale CAS, reopened upstream | BLOCKED | no dispatch |

Handoff prompts include the prior stage's exact verified artifact references. A CHECKPOINT never stands in for FINAL. The explicit GO PUBLISH text is NOT an authorization, and all planner outputs set publicationAuthorized=false and mutationPerformed=false.

## Verification

Tests: tests/equity-v3-pilotage-adapter.test.ts

Expected CI: npm run lint, npm run typecheck, npm test, PostgreSQL migration matrices and npm run build, on the isolated review branch.

No contract rewrite, no database migration, no production deployment, no workflow dispatch, no merge and no new HTTP entrypoint. VNext modules are not merged; their runtime authority differs from main.

## Required subsequent implementation

A separate authorized phase must implement authenticated Registry read models and real verified artifact loading, replace untrusted gate flags with verified manifest-derived facts, bind actual stage transitions to server-only RPC calls with CAS/idempotency, and use a wholly separate explicit publication-authorization path. No production activation may be inferred from a green CI run alone.
