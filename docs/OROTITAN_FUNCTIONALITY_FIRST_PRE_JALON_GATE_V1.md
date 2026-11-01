# OROTITAN — FUNCTIONALITY FIRST / PRE-JALON DESIGN GATE — V1 CANDIDATE

Date: 2026-11-01. Scope: engineering milestones J1–J8; SHADOW ONLY.
Status: **implementation candidate** (not a frozen analytical contract; not a production feature).
User engineering instruction: **functionality first, simplest compliant solution, no exploratory overengineering**.

## Non-negotiable objective

**Before every milestone's build, prove what functional result will work, how it will be tested, and why the selected implementation is the simplest compliant option.** Research/review the current implementation and official documentation first. Prefer existing tables and handlers. A short research spike is justified only to reduce a specific uncertainty. No speculative microservices, agents, migrations or abstractions.

Two independent gates remain distinct:

1. **Entry:** predecessor formally accepted by the persisted C8 decision, no unresolved predecessor incident; a pinned, reviewed, minimal design plan with all eight existing exit criteria; and separate, explicitly verified `GO BUILD Jn SHADOW`.
2. **Exit:** Jn's eight existing criteria, incidents and independent C8 human acceptance. Plan approval is **not** milestone acceptance. GO BUILD is **not** GO RESEARCH or GO PUBLISH.

The initial roadmap `initial_status` is historical and **never** demonstrates acceptance. Engineering gates do not alter the Research/Deep Dive/Integration methodology, stage admission, scores, publication rights or analytical `GO` procedure.

## A single one-page design decision, before coding

Document the following, preferably in one Markdown page:

1. **Result:** the user's observable operation and pass/fail acceptance example.
2. **Baseline:** exact current code, Registry state, blockers, contract pins, existing components worth reusing.
3. **Eight exit criteria:** exact text, unchanged; expected proof for each.
4. **Alternatives:** reuse-minimal default versus at most two credible alternatives; why the selected choice is simpler and still safe.
5. **Implementation:** smallest end-to-end vertical slice, followed by only necessary resilience/security work.
6. **Tests:** normal path, denial without authorization, duplicate/replay, open blocker, persistent readback, rollback. Adjust to the exact jalon, retaining all relevant mandatory guards.
7. **Stop-loss:** conditions triggering immediate redesign: contract drift, unidentified environment, security breach, unexpectedly multiplying exceptions, regressions or cost/complexity greater than the value delivered.
8. **Independent challenge:** reviewed exact document SHA-256, independently verified reviewer identity, verdict and objections. Review is **not** evidenced merely by two different free-form labels.

Version the one-page Markdown at a 40-character GitHub commit, record its exact SHA-256, and reject review if the content changes. A revision creates a new version and new review; never overwrite the previous decision. If prerequisites cannot be proved, output **BLOCKED** or **REVISE_PLAN**, not an invented approval.

## Delivery discipline (small, observable, reversible)

For each implementation slice: **expected behavior → minimal change → reproducible positive and negative tests → artifact/receipt → review → continue or stop**. Keep a working integration path as early as possible. Exclude speculative features explicitly. A completed plan does not allow touching production, publishing, accepting J0 or starting J1.

## Current implementation boundary

The small `pre-jalon-readiness-v1.ts` function computes a **fail-closed, read-only** engineering readiness projection. The SHADOW cockpit displays J1–J8 blockers from the actual persisted C8 decisions and the existing journal. The first tranche deliberately has **no plan approval/write API, no new table, no DB migration, no automatic build transition**. Plans, independent identities and GO may only later be supplied by independently verified **server-side** authoritative evidence; the UI must never supply boolean attestations as authority.

The pure evaluator is **not by itself enforcement of every future build endpoint**. Any future Jn start handler MUST call a trusted server guard that derives these inputs independently; until that has been implemented and tested, **no start action is enabled**. The UI labels alone are not an authorization decision.

**Current gate outcome:** J0 has not been accepted; one C8 incident remains open; J1 is **DRAFT_ONLY / NO-GO BUILD**, Research and publication remain prohibited. Existing historical `J1=EN_COURS` is not a permission.

## Regression criteria

- J0 unaccepted or predecessor incident open means J1 can only be drafted, even with a claimed plan or GO.
- Eight criteria required; invalid or incomplete plan fails closed.
- Unverified source hash, stale review hash, unverified reviewer or same author/reviewer denies plan.
- Reviewed approved plan without distinct `GO BUILD` never authorizes construction.
- `GO BUILD` for the wrong milestone or an unverified reference is rejected.
- No transition, C8 decision, production mutation, analytic stage or publication occurs from this feature.

## Sources of authority (unchanged)

- `OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0`
- `OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V1_FREEZE_V1.0.1`
- `OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0`
- Existing J0–J8 roadmap and C8 acceptance contracts

This is an engineering operating rule subordinate to the frozen methodology, not a change to those documents. No extra micro-validation gates are added to the analytical Research → Deep Dive → Integration flow.
