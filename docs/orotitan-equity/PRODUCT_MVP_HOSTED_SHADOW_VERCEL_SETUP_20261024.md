# OroTitan Hosted Research — Vercel isolation inventory / first deployment blocker (2026-10-24)

**Status: ISOLATED VERCEL PROJECT READY; RESEARCH HOSTED URL NOT DEPLOYED.**
No Research executed, no new Registry run, no production Supabase writes, no PR merge or GO PUBLISH.

## Project created, verified

- Team: `OROTITAN` / `orotitan` (`team_Y0LNO6zSfzsYNODysA4sRpms`), plan **Hobby**.
- New *separate* project: `orotitan-research-shadow-preview`, ID `prj_RBi6Sw0XcL0AbxSHVfoTwDkyQAHE`.
- Framework Next.js; Node 22.x.
- Vercel Authentication / Deployment Protection **all** enabled and verified on the new project. Not an edit to the existing production project.
- No Git repository connected to this new project on creation; deploy attempts used an explicit SHA and review branch.
- No billed add-on or paid Vercel build machine enabled.

Only these non-secret variables were created, **Preview-only**:

| Key | Value |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://awgsurdyvsyolcgpnygh.supabase.co` |
| `OROTITAN_SHARED_INBOX_TEST_URL` | `https://awgsurdyvsyolcgpnygh.supabase.co` |
| `OROTITAN_SHARED_INBOX_TEST_PROJECT_REF` | `awgsurdyvsyolcgpnygh` |
| `OROTITAN_EXPECTED_SUPABASE_PROJECT_REF` | `awgsurdyvsyolcgpnygh` |
| `OROTITAN_SHARED_INBOX_TEST_ENABLED` | `1` |
| `OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED` | **`0`** (intentionally disabled) |

**NOT configured**: `SUPABASE_SERVICE_ROLE_KEY`, `OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY`, `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`. No values copied from existing Vercel projects, which may contain production credentials. The new project cannot operate the Research inbox without these independently provisioned **SHADOW-only** secrets.

## Why the first deployment was canceled

An explicit Git SHA deployment of review branch `review/orotitan-product-hosted-shadow-preview-gate-20261024`, SHA `af98f52e475a196f47c150ba89b08e176e728325`, was requested on the new isolated project **with `target` omitted**, documented by Vercel API to default to Preview.

However, Vercel immediately classified the **first deployment** as `target: production`. ID: `dpl_Bqkxv4vBJw9WUsCywPushhkuS6EU`. This was **canceled immediately** before acceptance. Subsequent inspection confirms `state=CANCELED`, `target=production`, no READY deployment and no usable demo URL.

The project had `ssoProtection: all`, no production Supabase credentials, and the hosted Research enable flag is Preview-only at `0`. The code independently rejects Vercel `VERCEL_ENV=production`. No backend Research request was sent.

This is consistent with a documented Vercel first-deployment target-classification issue: GitHub vercel/vercel issue #17069 (July–September 2026); Vercel staff have acknowledged the first deployment can be automatically classified Production. See https://github.com/vercel/vercel/issues/17069.

**Do not retry a targetless/staging deployment blindly on the empty project.** Do not treat the canceled target=production deployment as successful Preview. An intentional *isolated, fail-closed bootstrap* production deployment would be a different, separately reviewed action and must not be confused with an OroTitan main-site release.

## Remaining operator actions / acceptance gates

1. Choose a provider-supported first Preview deployment path or explicitly authorize a fail-closed bootstrap only on this dedicated isolated project. Verify actual returned `target=preview` **before** claiming success.
2. In **this new Vercel project's Preview environment only**, configure **SHADOW** service-role credential in `SUPABASE_SERVICE_ROLE_KEY` and `OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY` (identical value), and a dedicated `ADMIN_PASSWORD` and random `ADMIN_SESSION_SECRET` (>=32 chars). Use Vercel environment secret UI; do not paste secret values in GitHub, ChatGPT, build logs, or client-exposed `NEXT_PUBLIC_` keys.
3. Once isolated secrets and exact SHA are verified, set `OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED=1` in **Preview only** and initiate a single reviewed Preview deployment.
4. Validate authenticated Next.js HTTP requests to SHADOW PostgREST, then a browser on a genuine phone and independent computer; the identical durable request UUID after refresh and duplicate submission must be observed.
5. Repeat SQL gate: one receipt per synthetic company, `registry_run_id=NULL`, `orotitan_runs` and artifacts unchanged, `anon`/`authenticated` denied, admin auth enforced.
6. Keep PR #419–#422 draft, no main merge or GO PUBLISH.

## Boundary / outcome

Achieved: live SHADOW PG17 schema and receipt, SQL deduplication, safe isolated Vercel project with all-deployments SSO, controlled Preview-only nonsecret configuration, successful CI for PR #422.

Not achieved: a valid Preview deployment URL or cross-physical-device application acceptance. **Do not mistake infrastructure scaffolding or canceled deployment for a functional hosted Research pilot.**
