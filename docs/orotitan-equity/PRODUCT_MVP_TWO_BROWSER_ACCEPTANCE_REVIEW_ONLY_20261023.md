# Product MVP 03 — Real two-browser acceptance of shared Research inbox

**2026-10-23, DRAFT review only.** Stacked on product MVP #420; no cloud
migration, environment provisioning, Merge, deploy, OAuth writes or Registry work.

## What this actually tests

A disposable GitHub Actions Ubuntu runner starts:
1. A clean PostgreSQL **17** database with existing Screener schema and the
   reviewed **nonproduction** Research inbox SQL, seeded with two synthetic companies.
2. A tiny **test-only** HTTP service bound strictly to
   `127.0.0.1:48761`, translating only the specific PostgREST requests needed
   by the existing Supabase SDK into that disposable PG17 fixture.
3. The actual Next.js development server at `127.0.0.1:3184`, with a
   hard-gated `CI=true`, `NODE_ENV=development`,
   `OROTITAN_PRODUCT_BROWSER_FIXTURE=1` configuration. **Never** hosted.
4. Two independent **Chromium browser contexts**, one using a 390px mobile
   viewport and one a 1420px desktop viewport, plus an unauthenticated
   third context for redirect access control.

The test logs into the **actual admin form** separately from mobile and
desktop, checks signed HTTP-only cookies, opens the actual
`/admin/research-requests` React page, selects an actual synthetic row
served by the disposable Screener database, submits a Research request
from the mobile browser, then reloads the desktop browser and confirms
that it observes the **same persisted UUID** via the shared database.
Submitting the same company from the desktop browser returns the original
receipt and `result=existing`, rather than creating a duplicate.
The mobile page reload confirms only one request. PostgreSQL is queried
again to assert a single row and absence of the OroTitan Registry table.

## Separation from live Supabase

The loopback HTTP service **only emulates the narrow PostgREST API endpoints
under test**. It is NOT an actual Supabase deployment or a broad PostgREST
implementation. Specifically, this test **does not** validate real hosted
Supabase Auth, JWT, RLS enforcement via PostgREST, platform quotas,
network ingress, CDN cookies, real phones, iOS Safari, or actual browser
connectivity to a hosted project.

The test-only client escape hatch requires ALL conditions: genuine Next
development mode, `CI=true`, explicit local browser fixture flag,
fixed `http://127.0.0.1:48761` endpoint, the exact local fixture
reference, and a synthetic test key of adequate length. It does not
allow untrusted URLs, external hosts or any production credential.
The ordinary shared-test Supabase URL allowlist and hard denial of the
OroTitan production project remain unchanged outside this fixture.

Playwright Chromium is installed only on the disposable CI runner using
`npm install --no-save --no-package-lock`; this PR does **not**
change the application production dependency list or lockfile.

## Product acceptance and next gate

If the separate CI job is green, we have demonstrated two independent
**real browser sessions** using the actual Next admin pages and one
disposable PostgreSQL backend. This proves local product **web flow**,
not a hosted or device-to-device deployment.

The next go/no-go concerns enabling one authoritative shared queue in
a real nonproduction or approved hosted environment and verifying
real phone+PC persistence. Before any cloud writing, independently
review administrative authorization, credential isolation, schema/RLS,
request-lifecycle/renewal, canonical issuer-security mapping and migration
impact. The D30–D38 PRs must be triaged rather than blindly merged.

**User authorization boundary:** no active Supabase Free migration, paid
plan, Vercel deployment, main merge, data changes, MCP writes, Research
run creation, evidence admission or Besi stage progression.
