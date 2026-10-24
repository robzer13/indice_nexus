# OroTitan Research Intake — Pre-run Routing Card V1 (REVIEW ONLY)

Date: 2026-10-25
Previous preview: PR #423, deployed, independently accepted on real desktop + Android mobile.
Current proposal: read-only pre-run preparation card. **No execution, GO, RUN_ID, schema mutation, production deploy, publication or contract change.**

## Governing authorities and precise boundaries

- `OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V1_FREEZE_V1.0.1`, §§4–10, requires issuer/security identity, canonical dossier state, entry path, ANALYZE/INITIAL or REFRESH choice and a concise pre-run card. A separate `GO <COMPANY>` authorizes execution, and must occur **before** a persistent RUN_ID. A different `GO PUBLISH` is needed for publication.
- `OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0`, §§4–5, keeps Pilotage orchestration-only; `DATA_CUTOFF` is set to user-specified historical date or to run initiation date. Do not pre-set to preview-visit time.
- `OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0`, §§5–6, requires creation idempotency, immutable cutoff, immutable pin set and durable contract locators / content hashes. Contract-pin verification is **NOT implemented** here and must not be marked done.

## MVP behavior

1. The existing Research request stays `REQUESTED_NOT_EXECUTED` with `registry_run_id = NULL`.
2. A strictly isolated, authenticated Vercel SHADOW Preview server makes read-only queries to `issuers`, `research_dossiers`, `research_snapshots`, `orotitan_runs`, using existing `securities` identity candidates.
3. Exact **ticker AND exchange** match is needed for a candidate. A candidate does NOT confirm issuer/security legal identity. Missing legal name/country remains flagged.
4. For exactly one candidate issuer:
   - no issuer dossier at all → **propose** `IMPOSED_COMPANY / ANALYZE / INITIAL`, subject to identity + alias verification;
   - exactly one active dossier with a consistent `current_snapshot_id`, joining `research_snapshots` by exact dossier and issuer → **propose** `REFRESH / REFRESH` with the baseline UUID;
   - historic inactive dossier, multiple active dossiers, missing/mismatched snapshot → block mode proposal.
5. Existing nonterminal Registry runs receive an explicit review blocker. The card does not assume a new run is always needed.
6. Unavailable/truncated/malformed Registry data → fail closed as **unverified**, never silently treat as no existing dossier.
7. The card shows proposed routing, blockers, immutable contract-pin status **unverified**, DATA_CUTOFF **unset**, execution GO **not received**, publication GO **not received**, and `executionAdmitted=false` regardless of proposal.
8. No server action for execution authorization, no GO input/control, no Registry mutation, no analytical computations. The current frozen methodology `Evidence → Calculation → Judgment → Narrative` remains unchanged.

## Evidence and known unknowns

The non-production SHADOW Registry contains independent synthetic securities such as `RC1`, `RC1V1` and `RC1V2`, but the hosted Research request fixtures are `TAL / XTEST` and `TBE / XTEST`. There are no exact security matches for these requests. Consequently the expected two hosted pre-run cards must show **identity unresolved, mode unproposed, execution blocked**. This is an intentional honest negative path, not a failure of the card.

In real companies, full legal identity and issuer/security mapping, an authoritative scan for aliases/duplicate dossiers, external evidence checks, an explicit user mode override where appropriate, cutoffs and verified immutable contract archives are still outstanding. None are implicitly certified by a database join.

## Acceptance criteria

- Read-only pre-run card visible per request on phone and desktop in authenticated test Preview.
- Synthetic Alpha and Beta both show *BLOCKED* with no issuer or mode inferred.
- Unit tests cover INITIAL and REFRESH proposals, inactive dossiers, malformed baselines, duplicate dossier ambiguity, unknown service state, outstanding RUNs, wrong exchange, invalid receipt and unchanged inputs.
- Disposable PostgreSQL 17 + independent Chromium mobile/desktop contexts remain the CI baseline.
- `anon` has no Registry write grant and no frontend action can create `orotitan_runs`.
- SHADOW post-deploy request count, run count and artifact count remain unchanged from 2 / 4 / 5. PR remains draft, no main merge.

## Next, not part of this PR

A new independently reviewed design must make identity confirmation durable and auditable, pin exact versioned contracts with durable content hashes, establish a validated canonical baseline (or prove none), and record the user’s **exact company-specific GO** as a distinct authorization event. Only then may a controlled idempotent Registry `create_run` function be considered. GO in this ChatGPT conversation to continue *design work* is not execution authorization for any company.
