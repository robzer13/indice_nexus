# OroTitan Hosted Research — Vercel isolation inventory / first deployment blocker (2026-10-24)

**Status update (2026-10-24): LIVE ISOLATED VERCEL PREVIEW READY; physical device acceptance NOT YET VERIFIED.**
No Research executed, no new Registry run, no production Supabase writes, no PR merge or GO PUBLISH.

## Real desktop + phone acceptance evidence (2026-10-24)

The operator manually used the protected SHADOW Preview on a **real desktop browser and a real mobile browser**, submitted the synthetic Beta Research request, and provided screenshots from both devices.

**Directly observed from screenshots**
- Desktop: Research inbox displayed two requests (Synthetic Research Alpha and Synthetic Research Beta), both `Demandé · non exécuté`; pilot run counter `0`. The desktop URL included `result=existing` for Beta and returned request UUID `b20f2252-5b5d-4ccc-b24e-f1d7b26a4dce`.
- Phone: the same two companies were visible with matching dates and times, total requests `2`, not-executed requests `2`, pilot run counter `0`.
- Thus **physical cross-device shared-list retrieval and rendering is evidenced**. This is not merely two emulated Chromium contexts.
- **Remaining proof gap:** the mobile screenshot shows only company/date columns; mobile UUIDs/status columns were beyond the horizontal scroll. Strict per-device UUID equality is **not visually evidenced** yet. User can swipe horizontally on the request table to expose the References.
- **Mobile UX observation:** request table overflows horizontally and requires a horizontal swipe to see the request UUID. Consider a separate responsive request-card layout in a later review PR; do not silently alter the validated deployed SHA.

**Authoritative Supabase SHADOW read-back after the screenshots** (SELECT only) returned exactly:
- `Synthetic Research Alpha`: request UUID `29402dfe-3f7d-43a1-bb08-e0b6c001141a`, status `REQUESTED_NOT_EXECUTED`, `registry_run_id = NULL`.
- `Synthetic Research Beta`: request UUID `b20f2252-5b5d-4ccc-b24e-f1d7b26a4dce`, status `REQUESTED_NOT_EXECUTED`, `registry_run_id = NULL`.
- Prior run/artifact count in this acceptance was 4/5; the application itself reports no Research runs created by this pilot. Do not assert these counts are freshly re-audited from this screenshot alone.

**Acceptance disposition:** hosted SHADOW Research request inbox is functional on real desktop and phone and shares the same request set. Strict physical-device UUID parity awaits mobile References-column inspection. This does **not** authorize any Research execution, Registry run creation, GO PUBLISH or production merge. PR remains draft.

---

## Update: isolated nonproduction deployment READY (2026-10-24)

The following observations **supersede the historical first-attempt blocker below**:

- User added Preview-only `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`, `SUPABASE_SERVICE_ROLE_KEY`, and `OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY` via Vercel secure environment UI. Their secret values have not been retrieved or recorded.
- Inspection found mistakenly duplicated Production-scoped entries. These were neutralized and user deleted them. A final inspection showed **zero Production-scoped environment variables** and all ten expected variables scoped to Preview.
- Protected, isolated **staging** bootstrap, target explicitly `staging`, deployment ID `dpl_D9FrstymU5jTohfGR6qNoSY4vbwu`, became `READY`. No bootstrap `production` deployment was needed or authorized.
- After bootstrap, explicit `target=preview` was rejected by Vercel API 400 (only `production`, `staging` or custom target string allowed). A subsequent targetless Git deployment correctly returned **`target=null` (Preview)** rather than Production, and became `READY`.
- **Research Preview** ID `dpl_3ZmpJtxaZb6uEAm7carxvEGiYBat`, target `null` (= Vercel Preview), commit `1354319219c99695239a0f24ed2bc8ffd5444310`, protected test URL:

  `https://orotitan-research-shadow-preview-ig6b7z2r5-orotitan.vercel.app/admin/research-requests`

- `OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED=1` was set **only on this separate project's Preview env** before deploying. The code independently rejects Vercel Production, arbitrary Supabase refs and missing/unequal service keys.
- Project SSO / Vercel Authentication remains **all deployments**, and `orotitan-research-shadow-preview` remains separate from the production `orotitan-screener` project. Project `live=false` at inspection (no new public production site).
- Real hosted Shadow SQL verification following deployment: **2 companies, 1 request (`REQUESTED_NOT_EXECUTED`, `registry_run_id=NULL`), 4 existing Registry runs, 5 artifacts**. This verifies stable DB state, **not HTTP application behavior**.
- The code's GitHub Actions two-browser Chromium/PG17 and Screener CI checks were previously **SUCCESS**. Their fixture uses a disposable local database, **not** hosted Shadow API.
- The protected hosted URL could not be accessed by the server-side browsing tools; it still requires **manual authenticated acceptance on a real phone and separate computer**. Do not claim web UI or cross-device acceptance until the user tests both.

**Acceptance procedure**: Login to Vercel Deployment Protection on both devices using an eligible team account; then enter the separately configured `ADMIN_PASSWORD` in app admin. Open `/admin/research-requests`; check existing `Synthetic Research Alpha` receipt; submit `Synthetic Research Beta` once; record the returned UUID without disclosing cookies or secrets; refresh and navigate from a second physical device to confirm the same persistent UUID. Re-submit Beta and verify deduplication; check SQL that no new `orotitan_runs` or artifacts were created. Remain entirely nonproduction.

---

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
