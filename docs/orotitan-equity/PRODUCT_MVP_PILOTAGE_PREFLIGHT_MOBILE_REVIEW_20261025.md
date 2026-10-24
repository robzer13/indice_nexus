# OroTitan Research MVP — Pilotage preflight + mobile receipts (review proposal, 2026-10-25)

**Status: REVIEW-ONLY PR; deployed to isolated, SSO-protected SHADOW Preview for UI acceptance. NOT merged to main. NO Research execution or GO PUBLISH.**

## Deploy / acceptance update (2026-10-25)

- Successful CI on commit `e3bca8af104de7ccca46248a30d8f85cdbfe53bb`:
  - Screener CI run `40137688094` = SUCCESS (lint, typecheck, unit/PG migration tests, build).
  - Research Two Browser Acceptance run `40137688079` = SUCCESS (isolated PG17 + Chromium mobile/desktop contexts).
- New protected Vercel Preview deploy `dpl_4LUHSo3DQerPF6wTk1mNeeLGVD6C`, commit `e3bca8af104de7ccca46248a30d8f85cdbfe53bb`, `target=null` (Preview), `readyState=READY`.
- URL: `https://orotitan-research-shadow-preview-npkellb16-orotitan.vercel.app/admin/research-requests`
- Target project `prj_RBi6Sw0XcL0AbxSHVfoTwDkyQAHE`, protected by Vercel Authentication for **all** deployments, no Production-scoped env variables at pre-deployment inspection; Next.js Shadow Preview enable gate remains mandatory.
- Supabase SHADOW post-deploy audit (read only): exactly `2` Research request receipts, `4` existing Registry runs, `5` existing artifacts. New code has no run-creation or mutation path.
- **Physical mobile acceptance of the NEW UI completed via operator-supplied screenshots (2026-10-25).** Mobile request cards showed both persistent Alpha and Beta UUIDs, matching the authenticated desktop view. A separate screenshot showed the Beta preflight expanded, with exact ticker/exchange Registry candidate count **0** and blockers: no matching security, canonical mode + dossier unresolved, missing DATA_CUTOFF, contract/artefact references not verified, and explicit GO missing. The displayed verdict was `BLOCKED_NOT_AUTHORIZED`. Alpha preflight remained collapsed in the submitted screenshot: only its blocked summary, not its expanded explanations, was observed.
- **Read-only SHADOW re-audit after mobile acceptance:** 2 request receipts, both `REQUESTED_NOT_EXECUTED` and `registry_run_id=NULL`; 4 pre-existing Registry runs and 5 artifacts unchanged. This supports no new Research execution.
- **Acceptance disposition:** hosted cross-device Research receipt continuity, responsive mobile UUID cards, and expanded Beta read-only preflight UI are accepted for the synthetic pilot. This does **not** approve execution functionality, real-world issuer resolution, stage admission, Registry run creation, contract verification or GO PUBLISH.



## Authoritative contract boundaries

Frozen authority:
- `OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0`: §4 Pilotage resolves issuer/security, mode, canonical state and requests a separate `GO <COMPANY>`; §5 a persistent RUN_ID exists **only after GO** and before Research; the data cutoff is run-specific.
- `OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V1_FREEZE_V1.0.1`: §§7–9 require a pre-run card, explicit execution GO and persistent Registry run state after GO; §28 failures fail closed.
- `OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0`: §3 requires RUN_ID, issuer identity, mode/run type, DATA_CUTOFF, contract pins, baseline where applicable, and no stage blockers; without these, do not Research.
- `OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0`: idempotent run creation after explicit GO, not from a Research request; separate GO PUBLISH.

The MVP's `public.orotitan_research_intake_requests` receipts remain immutable and **always `REQUESTED_NOT_EXECUTED`, `registry_run_id = NULL`**. No schema or Registry changes are proposed by this PR.

## What the new code implements

1. A pure, read-only `assessResearchIntakePreflight` function which always returns `BLOCKED_NOT_AUTHORIZED` until a future, separately reviewed execution authorization implementation exists.
2. The server, **only when hosted SHADOW Preview guard validates**, reads candidate rows from `public.securities` by ticker. The pure function then requires both exact ticker **and exchange**, counts candidates and blocks any missing, ambiguous, or merely provisional identity match. One match is a **candidate**, never issuer confirmation.
3. Unavailable/malformed/truncated Registry reads are treated as `REGISTRY_LOOKUP_UNAVAILABLE`, not false zero matches.
4. Independent blockers remain for canonical mode + prior dossier/snapshot, DATA_CUTOFF, contract pins and missing explicit execution GO. There is **no GO or run-creation button**, no call to `orotitan_runs` INSERT, no new statuses on request receipts, and no server action to create a run.
5. Desktop preserves the existing request history table; mobile replaces horizontal-scrolling rows with readable stacked receipt cards including UUID and status. The expanded read-only Pilotage panel lists blockers per request.

## Explicit non-features

- This is **not** issuer identity resolution or a canonical issuer mapping from the Screener.
- This does **not** choose `INITIAL` vs `REFRESH` or claim a canonical baseline.
- This does **not** record user GO or compute DATA_CUTOFF.
- This does **not** create an `orotitan_runs` row, `RUN_ID`, artifact, stage transition or publication.
- Matching a ticker alone does not grant eligibility; a share/security is not an issuer.
- No production Supabase operation, deployment promotion or draft PR merge.

## Acceptance checks

- Unit negative tests: no candidate, exact single candidate (still blocked), duplicate candidate (ambiguous), mismatched exchange, unavailable Registry, bad existing request status/run id, input immutability and blocker explanations.
- Disposable PostgreSQL 17 + independent phone and desktop Chromium browser contexts. E2E explicitly checks mobile UUID is visible without table horizontal scrolling, read-only Pilotage BLOCKED state and no execution action.
- Existing CI Screener workflows remain required. No live SHADOW Supabase writes from new test code.

## Next implementation milestone — separate explicit authorization phase

A *future* proposal may persist a versioned pre-run card with a real issuer/security identity, inspected dossier and snapshot, canonical mode/run type and contract pins; obtain the user's exact GO; then create a durable Registry run with a deterministic `run-create:<client-intent-id>` idempotency key and concurrency control. That implementation requires new independent review and user authorization, and must not bypass Research preflight or any freeze contracts.

## Hosted Preview deployment boundary

The earlier PR #422 protected Preview URL remains a distinct historical deployment. The new protected Preview URL and pinned SHA are listed in the deployment update above; this PR remains **DRAFT** and has not been merged. A later opt-in shadow-only Preview redeploy must verify project ID `prj_RBi6Sw0XcL0AbxSHVfoTwDkyQAHE`, exact SHA, SSO all, all Preview-only secrets, no Production env, strict SHADOW ref `awgsurdyvsyolcgpnygh`, and post-deploy absence of new Registry runs/artifacts.
