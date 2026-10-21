# D29a — OAuth Pilotage status, never an unverified execution prompt

**Status:** PROPOSED / NOT DEPLOYED  
**Base:** exact deployed pilot commit `1e6296c89ae213827b3165e162bac2c8afbc75db`  
**Project:** `orotitan-chatgpt-mcp-pilot` only  
**Contracts:** frozen Pilotage V1.0.1, Research Execution Process V1.0, Research Stage Contract V1.0, Registry Manifest Specification V1.0.  
**No database, RLS, OAuth client, Registry, contract or publication changes.**

## What the real pilot can do

- OAuth consent, JWT/issuer/audience/resource verification and ChatGPT tool discovery are already live.
- D28's separately authorized, Besi-scoped SELECT GRANT + RLS was installed and `get_orotitan_run` returned the Besi `RESEARCH` blockers.
- The OAuth `get_orotitan_next_prompt` endpoint currently returns `DELEGATED_PROMPT_READ_DISABLED`, because the D2 proof reader is privileged and has broader data access.

## D29a boundary

When the caller is authenticated with Supabase OAuth:
1. Re-use the **existing request-scoped JWT**, explicit permitted run-ID check, Supabase publishable key and RLS-filtered `getMcpRunContextDelegated`.
2. Return only *observed* Registry status, current stage, exact blocker summary, state versions and cautious remediation guidance. The tool's name remains compatible with ChatGPT's discovered action.
3. **Always return** `prompt:null`, `verifiedPrompt:null`, `nextStepAuthorized:false`, `verificationPerformed:false`, `publicationAuthorized:false`, `mutationPerformed:false` and `trust:REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION`.
4. If stage data is missing, unsupported versions or the run is nonexecutable, fail closed. A Registry `FINAL` manifest or `YES` handoff gate never becomes a verified executable prompt.
5. Do not call the privileged D2 immutable-artifact loader for OAuth. Do not expose storage, GitHub private artifacts, service-role credentials, or other run IDs. The quarantine tool remains disabled for OAuth.

The server's non-OAuth trusted D2 path is preserved and retains its independent immutable-evidence authority. D29a does **not** make it public.

## Distinction from D29b

D29a is a bounded **operational blocker/status** enhancement, not yet full verified prompt retrieval. For *D29b*, design a separately reviewed immutable-artifact verification mechanism using per-subject, per-run, per-artifact authorizations and explicit contract/hash checks, rather than routing OAuth JWTs into existing service-role D2 readers. Require proof completeness and CAS/version checks before ever releasing an execution prompt. D29b is outside this PR and requires a separate review, tests and deployment authorization.

## Tests

`tests/equity-v3-mcp-delegated-pilotage.test.ts` covers:

- current Besi Research blocked state + exact blockers;
- apparently final/YES Registry state (still no verified prompt);
- invalid process/contract versions and nonexecutable run;
- missing stage observations;
- static OAuth routing isolation from the D2 privileged reader.

All tests are unit/source tests using fixtures. They are **not** an OAuth production token or API penetration test.

## Rollout gate

1. GitHub CI lint, typecheck, unit/PG17 tests and both builds must succeed.
2. Diff must contain only this D29a code/tests/docs, and no database migrations, secrets, RLS, registry or contract changes.
3. Obtain explicit, separate authorization for **only** `orotitan-chatgpt-mcp-pilot` deployment at an exact tested SHA.
4. Re-run the real OAuth tool call on Besi; verify status and blocker observation, no executable prompt; negative test on a different run; check no write tool is enabled.
5. If any condition fails, stop. No automatic rollout to `orotitan-screener`.
