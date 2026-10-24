# Product MVP 04 — Hosted Supabase SHADOW Research intake proof (2026-10-24)

**Status: SQL-hosted acceptance PASS; actual phone/PC browser acceptance NOT TESTED.**  
**Environment:** strictly nonproduction `orotitan-vnext-shadow` (Supabase ref `awgsurdyvsyolcgpnygh`), PostgreSQL 17.6 / region eu-west-3.  
**Production ref hard excluded:** `cugpgtzygqqlxetyeven`. No prod migration, write, deployment, or pull-request merge.

## Actual baseline
- Project `ACTIVE_HEALTHY` after restoring an existing Free-plan project. No new project or paid branch created.
- Pre-change `public` contained 15 tables. The catalog's approximate `rows:0` was **not a reliable row count**.
- Exact SQL baseline: 4 existing Registry runs, 5 artifacts, 3 issuers, 3 securities, 3 dossiers, 3 canonical snapshots; no legacy Screener companies. Existing issuer names were synthetic RC1 shadow fixtures. Existing Registry rows were **not modified**.
- Existing Vercel `orotitan-vnext-pilotage` PREVIEW `NEXT_PUBLIC_SUPABASE_URL` resolves to the *production* Supabase ref. Therefore using the existing preview credentials is **explicitly disallowed** for the Research pilot.

## Shadow-only physical changes
Two tracked **Supabase SHADOW-only** migrations applied through the Supabase management plane:
1. `20261024134009 product_research_inbox_shadow_nonproduction_v1`: creates `public.orotitan_research_intake_requests` and its validation/immutability triggers, using the candidate SQL from product PR #420, with an added fail-closed preflight requiring PostgreSQL 17, absence of the request table, zero Screener companies, and the synthetic RC1 shadow issuer marker. No Research Registry run is created.
2. `20261024134059 product_research_shadow_strict_service_acl_v1`: revokes service-role default full-table privileges and restores **SELECT/INSERT only**. The initial creation inherited `service_role` ALL through default privileges, even though the proposal only granted SELECT/INSERT. This was detected and corrected before data insertion.

Only two synthetic active Screener fixtures were inserted:
- `00000000-0000-4000-8000-000000000091` / `Synthetic Research Alpha` / `TAL` / `XTEST`
- `00000000-0000-4000-8000-000000000092` / `Synthetic Research Beta` / `TBE` / `XTEST`

One synthetic Research intake receipt was inserted **as `service_role`** and read again from a distinct SQL operation:
- request UUID: `29402dfe-3f7d-43a1-bb08-e0b6c001141a`
- company: `Synthetic Research Alpha`
- status: `REQUESTED_NOT_EXECUTED`
- `registry_run_id = NULL`

Separate-hosted-database-operation tests passed:
- Duplicate same-company insertion raises the expected unique violation; only one receipt exists and readback UUID remains identical.
- Unknown-company insertion raises `PRODUCT_INBOX_COMPANY_NOT_ELIGIBLE`; no row created.
- RLS enabled; direct table SELECT/INSERT grants to `anon` and `authenticated` absent.
- `service_role`: SELECT/INSERT = true; UPDATE/DELETE = false.
- Existing Registry run count unchanged: 4. Existing artifact count unchanged: 5.

## What is NOT proven
- No hosted PostgREST HTTP SDK request has been executed with the shadow service credential.
- No Vercel Preview has been deployed for Research inbox.
- No two real independent devices have visited the same hosted site; iOS Safari/mobile networks untested.
- Real hosted Admin login, cookie handling, browser-to-server authentication and web request deduplication remain unverified.
- `shadow` is nonproduction but **not disposable**. Do not reset or drop it; it includes other RC1 Registry acceptance fixtures.

## Release / isolation blockers
1. The implementation intentionally disables the Research inbox whenever `NODE_ENV !== development`. Vercel Preview ordinarily uses production-mode Next.js, so the current application **cannot** be safely enabled there by merely adding an environment flag.
2. Existing Vercel Preview envs reference production Supabase. Do not deploy this branch with those credentials. A dedicated preview environment/project must bind **all** app Supabase credentials to `awgsurdyvsyolcgpnygh`, with no production fallback, separately isolated admin secrets, SSO/protected access, and fail-closed preview-only gates.
3. Obtain/attach the SHADOW project's server-only API credential through a secure secret-management workflow; **never copy a production service-role secret**, place secrets in GitHub commits or chat messages, or expose them through `NEXT_PUBLIC_` variables.
4. Review actual hosted PostgREST Data API exposure/grants and run a negative authentication test before the browser demo.
5. Run a real phone + computer against the same isolated HTTPS application, compare stable request UUID after refresh and duplicate submission, then record results.

**Decision:** hosted PostgreSQL shared-storage gate = PASS. Real hosted web / cross-device gate = NOT VERIFIED. No production promotion, no PR merges, no Research execution, and no Besi stage advancement authorized.
