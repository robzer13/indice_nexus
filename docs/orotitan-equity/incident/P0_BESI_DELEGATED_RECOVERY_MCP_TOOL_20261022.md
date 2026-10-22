# P0 — ChatGPT delegated Besi recovery worklist (D29a pilot)

Status: **DRAFT / code-review only**. The pilot production deployment
`orotitan-chatgpt-mcp-pilot` currently serves Git SHA
`5740dc40022993f414da1f8397de0d7725533095` from
`feat/orotitan-d29a-oauth-safe-pilotage-20261021`. This change is branched
directly from that SHA and targets that **pilot-only branch** for review.
**Never merge its OAuth implementation into the Screener main branch by
accident.** Do not redeploy, alter RLS, grant new roles or run migrations
solely because this proposal passed CI.

## Implemented — read only

New OAuth MCP operation `get_orotitan_blocker_worklist` accepts exactly the
authorized Registry `run_id` UUID, with zero other inputs or user-supplied
instructions. The existing HTTP OAuth gate verifies bearer
signature, issuer, audience, resource, client and allowed subject. The tool
**requires delegated user JWT** (no static token mode fallback). It uses the
existing Supabase publishable-key RLS client and the exact run allowlist.
A failing Registry query or absent stage version returns a controlled error;
there is NO fallback to a `service_role` read.

The handler converts the observed stage and its bounded blockers into the
fixed, deterministic `buildRecoveryWorklist` classifier from the
independent (draft) pilotage PR #401. No LLM, no network retrieval and no
privileged evidence bytes are involved. Each OPEN_UNVERIFIED item contains:
- blocker as observed in the Registry (NOT verified evidence),
- a fixed, bounded research recipe and explicit contradictory-source step,
- minimum evidence needed for future closure,
- an escalation instruction for genuinely unavailable sources.

Output carries `runStateVersion` and `stageStateVersion`; stage or run
revision change requires a new lookup. It *always* returns
`mayAdvance=false`, `mayPublish=false`, `mayMutate=false`,
`nextStepAuthorized=false`, `verifiedResearchPerformed=false`,
`mutationPerformed=false`. Unsupported contract revisions fail closed.

The current Besi test fixture contains six observed research gaps. Tests
also check absent manifest, disallowed privileged calls, and no OAuth-to-
service-role bypass. Scope is limited to read-only diagnostic, not a signed
proof or an authorization to change the Registry.

## Deliberately not implemented or authorized

1. `list_orotitan_submissions` for OAuth still fails closed with
   `DELEGATED_QUARANTINE_READ_DISABLED`. This is not a transient failure
   to bypass using server secrets. Existing quarantine bucket cannot be
   opened merely by passing a run UUID in the input.
2. OAuth write operations remain entirely disabled: `mcpWriteEnabled`
   always returns false; there is no token scope that confers Registry
   or Storage mutation.
3. No stage transitions, artifact acceptance, CHECKPOINT replacement,
   SOURCE_LEDGER mutation, Research FINAL attestation or `GO PUBLISH`.
4. No autonomous external research is executed by the backend tool.
   ChatGPT can **separately** research public sources after reading the plan,
   but that alone never changes evidence admission or status.
5. No publication of private source packs in the public indice_nexus
   repository; its branches are all publicly readable.

## Required phase 2 — secure evidence path (review gate, no shortcuts)

- Build an **explicit run+stage+revision scoped capability** for authorized
  OAuth user `sub` and consented OAuth `client_id`, distinct from basic
  OIDC `email/profile` claims. Do not rely on user-editable metadata.
- Provide a minimal *read* route that returns only stage-matching,
  user-authorized **quarantine metadata** (never bucket-wide enumeration).
  Download of private original bytes requires an even narrower permission
  and audit record. No service-role fallback inside the public MCP request.
- Add *submit draft evidence* with strict max bytes, MIME allowlist,
  CAS `runStateVersion` and `stageStateVersion`, idempotent hash,
  run/stage provenance, immutable quarantine storage and server-side
  actor/client audit. The receipt must read `RECEIVED_NOT_VERIFIED`.
  A user GO to research is not an automatic GO to every write operation.
- Build independent D2 verification against **immutable bytes**, source
  provenance, original contract pins, ledger conflicts, canonical
  evidence SHA-256 and current Registry version. A model summary or
  receipt is not evidence.
- Only an approved privileged authorizer may create a new CHECKPOINT and
  reconcile events via a CAS transaction, then re-check admission
  independently. Never mutate a prior sealed artifact in place.
- On continuing insufficiency, expose a targeted five-field
  assistance request (document, reason, attempted sources,
  acceptable alternative, effect on admissibility).
- Exercise negative tests: wrong `run_id`, JWT client/subject, stale
  revision, missing manifest, counterfeit receipt, spoofed evidence,
  denied upload, stage transition attempt and downstream cross-run reads.

## Acceptance for this PR

GitHub Actions lint, TypeScript, unit/integration, build. Post-merge
deployment **only to isolated OAuth pilot** requires a separate production
authorization and a new OAuth connection/reload in ChatGPT to discover the
tool. Do not touch `orotitan-screener` or `orotitan-vnext-pilotage`.

## Registry baseline

Besi observed UUID:
`48b9426b-4d0d-4858-9d87-efcd13ac415d`; RESEARCH,
`BLOCKED_INSUFFICIENT_INPUT`, `READY_FOR_DEEP_DIVE=NO`,
run version 6 and stage version 4 at last check.
This observation can change; the tool re-reads on each invocation.
