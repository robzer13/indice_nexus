# Hosted SHADOW Research Preview — gated deployment checklist (2026-10-24)

**Status: CODE REVIEW ONLY. No Vercel Research deployment and no hosted browser acceptance until all gates pass.**

This draft PR stacks on Product MVP PR #421. It adds an explicit **fail-closed Vercel Preview-only** gate for the Research request inbox; it cannot create Research runs or publish analysis. It must never run against production Supabase.

## Hard target boundary

- Supabase nonproduction project: `orotitan-vnext-shadow` / ref `awgsurdyvsyolcgpnygh`.
- Supabase production project (denied): `orotitan-screener` / ref `cugpgtzygqqlxetyeven`.
- Ordinary Next `NODE_ENV=production` is expected even in Vercel Preview.
- **Only** Vercel `VERCEL_ENV=preview` with `OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED=1` may turn on the shared intake. Vercel production always returns `disabled`.
- Next.js production deployments, local mode and arbitrary Supabase project refs remain denied in this new hosted gate.
- Preview's *normal app* Supabase endpoint must also be exactly SHADOW; privileged keys for the normal app and the shared intake must be identical SHADOW keys. This rejects mixed production app / shadow intake config.
- `ADMIN_PASSWORD` and `ADMIN_SESSION_SECRET` must be independently configured. A test-only dedicated login is preferred; do not inherit or reuse production credentials.

## Required Vercel **Preview-only** configuration (not production)

Configure **only on a dedicated isolated deployment or branch-scoped Preview environment**. Inherit **no** production Supabase credentials. Never add secret values to GitHub, documentation, PR comments, CI logs, or a conversation.

| Variable | Required value / handling |
|---|---|
| `VERCEL_ENV` | Platform-provided `preview`; do not manually override |
| `NODE_ENV` | Platform `production` Next.js build |
| `OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED` | `1` |
| `OROTITAN_SHARED_INBOX_TEST_ENABLED` | `1` |
| `OROTITAN_SHARED_INBOX_TEST_PROJECT_REF` | `awgsurdyvsyolcgpnygh` |
| `OROTITAN_EXPECTED_SUPABASE_PROJECT_REF` | `awgsurdyvsyolcgpnygh` |
| `OROTITAN_SHARED_INBOX_TEST_URL` | `https://awgsurdyvsyolcgpnygh.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_URL` | Same shadow URL; never production |
| `OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY` | **SHADOW** server-side service key stored as a protected Vercel secret |
| `SUPABASE_SERVICE_ROLE_KEY` | Identical **SHADOW** server-side service key in Vercel secret storage; never production |
| `ADMIN_PASSWORD` | Dedicated strong test-only admin password |
| `ADMIN_SESSION_SECRET` | Independently generated >=32-character test-only signing secret |

**Access control:** retain Vercel Deployment Protection / team SSO, HTTPS, and in-app Admin login. Check the exact deployed preview URL and branch before disclosing to testers. Do not disable production protection or attach a production domain.

**No secret-access shortcut:** Supabase connector currently exposes publishable project keys but no **server-side SHADOW secret retrieval** action. No hosted credential has been placed in Vercel in this change. The user/admin must configure the SHADOW server secret directly in the platform's protected environment UI. Do not substitute a publishable/anon key.

## Mandatory pre-deployment inspection

1. Inspect target Vercel project/team; the existing `orotitan-vnext-pilotage` Preview `NEXT_PUBLIC_SUPABASE_URL` currently points to the production ref, and hence must **not** be used unchanged.
2. Verify exact Git commit, scoped Preview env values, no inherited production secret, SSO policy, dedicated admin password/session secret, and isolated SHADOW project ref.
3. Confirm `public.orotitan_research_intake_requests` present on SHADOW, RLS enabled, `anon`/`authenticated` no table privileges, `service_role` SELECT/INSERT only.
4. Run unit tests `npm test`, typecheck, build, disposable PostgreSQL tests, two independent browser-context acceptance in GitHub CI.
5. Deploy **Preview only** on isolated target after protections and required secret are verified.
6. Via authenticated mobile browser + computer browser, validate persistent UUID / same single request after refresh / dedupe / unauthorized access rejection. Capture observations, never secrets or authenticated cookies.
7. Confirm `orotitan_runs` and artifacts unchanged. No run execution, no GO PUBLISH, no prod promotion.

## Tests

The pure gate tests cover exact successful SHADOW Preview, mismatched/shadow/production URLs, revoked flags, Vercel production, non-Vercel environments, missing secrets, server-key mismatch and missing/weak admin configuration. Existing in-CI PostgreSQL and Chromium tests remain disposable and do not access hosted Supabase.

**Docker:** Currently GitHub Actions runs disposable PostgreSQL 17 as `postgres:17` service container; Docker installation on a user's computer is optional for these cloud workflows. Containerizing a future Research execution worker may be useful when reproducible isolated long-running computation is implemented, but it is not a prerequisite for this read/write receipt pilot.
