# P0 — Screener principal: production incident 2026-10-22 (review only)

## Scope and source of truth

**Production is the separate Vercel project `orotitan-screener`.** Its stable
alias `orotitan-screener.vercel.app` points to a READY production deployment,
GitHub `main` commit `9f102f521723483d1ddaa4803992dea855e5e4b1`
(`dpl_FkdsRZRtcm3jQ7kteFwe2sANU4e2`). Later `READY` preview
deployments must **not** be confused with the live production alias.

Supabase project `cugpgtzygqqlxetyeven` reported ACTIVE_HEALTHY, with 50
active dossiers and **34** active canonical snapshot pointers on 2026-10-22.

## Observed incident

Vercel runtime error groups across recent days:
- `/`: `Unable to load canonical dossiers: JWT issued at future`.
- `/admin.rsc`: `Unable to load market sync runs: JWT issued at future`.

Read-only Supabase edge-log evidence for `/rest/v1/research_dossiers`:
- **2026-10-22 07:48:03.163 UTC** HTTP **401**;
- **2026-10-22 07:48:03.285 UTC** HTTP **200**.
The 122-ms interval is consistent with a transient PostgREST JWT clock issue.
Do **not** infer that all requests fail or the database is empty.

Known upstream: Supabase/PostgREST sporadic `PGRST303: JWT issued at future`
reports include opaque `sb_secret_` server keys whose JWTs are minted at the
gateway. PostgREST v14.18/v16.3 documented corrections; verify the *actual*
hosted PostgREST rollout with Supabase Support rather than assuming those
versions are in use.
- https://github.com/orgs/supabase/discussions/48123
- https://github.com/supabase/supabase/issues/50651
- https://github.com/PostgREST/postgrest/releases

## Patch and safety boundaries

A narrow server-side Supabase `fetch` retry is proposed to restore GET/HEAD
reliability under *exactly* HTTP 401, JSON `code=PGRST303`, message containing
`JWT issued at future`, matching the configured Supabase origin and REST
`/rest/v1/` path. Max **two retries**, delayed 400ms and 1600ms.
Any other HTTP status / code / origin / endpoint, and all POST/PATCH/DELETE
operations, are untouched. Persistent rejection remains a 401, never
fabricated data or a successful fallback.

This is a **temporary resilience mitigation**, not a replacement for the
vendor-side clock/cache fix. Add a Supabase Support ticket with project ref,
region `eu-west-3`, failing UTC time, status-code pair, error grouping,
client type server-side and request IDs gathered privately. No private JWT,
`sb_secret_` key, user identity or credential should be included.

## Separate potential access blocker: Vercel SSO

Vercel project metadata shows `ssoProtection.enabled=true` with
`deploymentType=all_except_custom_domains`. The only discovered project
domain is `orotitan-screener.vercel.app`, which is not a user custom domain.
Thus public/non-team access to the Screener may also require Vercel
authentication. This access-control decision **must not be silently changed**:
first verify whether the intended Screener audience is private or public,
whether there is an external custom domain, and test access as that audience.
Do not switch off SSO as a workaround for PGRST303.

## Preconditions for production release

1. Review the read-only retry and negative tests; GitHub CI success.
2. Independently test public alias and the main `/screener` and admin
   paths as an authorized user; diagnose SSO separately.
3. Obtain explicit approval for the exact production `main` fix,
   merge only this isolated PR after review and verify the new Vercel
   **production alias** (not just a preview deployment).
4. Validate PGRST303 metrics after release and collect a user-visible
   smoke test. Continue escalating the vendor incident if persistent.

No D29b, VNext, OAuth pilot, RLS, Registry or publication changes belong in
this incident patch.
