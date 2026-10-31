# OroTitan J0 — C4/C5 verification and negative-test checkpoint

Date: 2026-10-31
Authority: user **GO J0 C4 + C5**, SHADOW only.
Target project: `awgsurdyvsyolcgpnygh`.
Excluded project: `cugpgtzygqqlxetyeven` (PRODUCTION).
Scope: validate governance journal and access controls. **Do not accept J0, start J1/Research, create a stage/snapshot, or publish.**

## C4 — Operational incident cycle: PASS for exercised scenario

- Open issue: `issue:j0-c4-live-20261031`
- `BLOCKER_OPENED` event UUID `35180c8a-96e6-4717-99e9-a869ca3bdb8e` (event_seq 8).
- Re-read persisted event; issue was open.
- Six live SHADOW negative cases, executed in `BEGIN/ROLLBACK`: duplicate opening denied, conflicting idempotency-key replay denied, closure of never-opened incident denied, forged milestone acceptance denied, direct UPDATE denied, direct DELETE denied. SQL matrix returned `J0_6_NEGATIVE_TESTS_PASSED`.
- `BLOCKER_RESOLVED` UUID `18bc62c1-33ac-4a4f-8825-593f4e066a60` (event_seq 9).
- Independent re-read confirms open and close persist in correct order. Issue is resolved (latest event: BLOCKER_RESOLVED).
- Criterion C4 evidence event UUID `16570f12-a702-4906-a8ec-72ff8414dc6e` (event_seq 10), `criterionIndex=4`. **Verification state REPORTED** in journal because the evidence reference is a tested issue, not an automatically re-read run snapshot. Separate SQL direct inspection confirms the issue lifecycle.

## C5 — Permissions and negative HTTP: PARTIAL, blocker open

### Verified directly on SHADOW

- Both `anon` and `authenticated` roles: no SELECT/INSERT/UPDATE on J0 events and no EXECUTE on J0 writer RPC.
- Direct SQL tests with `SET LOCAL ROLE anon` and `SET LOCAL ROLE authenticated` actually attempted restricted table reads and RPC invocation: both roles were denied.
- `service_role` can SELECT/INSERT and execute the controlled writer RPC, but cannot UPDATE the event journal. RLS is enabled. No table DELETE/UPDATE privilege for service role.
- The service-role negative matrix above confirms immutable events cannot be overwritten or deleted.

### HTTP application evidence

- A disposable Next.js 16 application build using **dummy, non-functional** local-only administrator and service secrets completed.
- Local application negative requests returned `401 UNAUTHORIZED` without a session and `403 ORIGIN_MISMATCH` with an invalid Origin when a synthetic **local-only** authenticated session was supplied.
- The local hostname/origin normalization prevented independent confirmation of the other intended `400` paths in that run; do **not** claim that full matrix passed.
- The real Vercel Preview is protected by Vercel SSO, which returns `401` at the edge before the OroTitan application. The temporary sandbox's Vercel CLI reported `Logged out`. Therefore application-level hosted Preview 401/403 was **not directly demonstrated**; no Deployment Protection was disabled and no bypass secret was created.

### SHADOW governance audit trail

- C5 evidence event: `58f6e940-aff2-4ab8-8796-f4a81a17b77f` (event_seq 11, `criterionIndex=5`, `REPORTED`, `allNegativesComplete=false`).
- Open blocking issue: `issue:j0-c5-preview-http-sso-20261031`
- Blocker event: `d931770c-5247-4734-8d32-86e90911a04e` (event_seq 12).
- This issue MUST stay open until a reproducible, duly authenticated test demonstrates the application responses on Preview and confirms no unintended writes. It is not sufficient to get Vercel SSO's own 401.

## Invariants independently re-read after C4/C5

- L’Oréal canonical run: `204390a1-5cba-4b9c-9821-ca0c7df43eca`: one row, `CREATED`, state_version 2.
- Stages = 0; run artifacts = 0; dossier snapshots = 0.
- Milestone ACCEPTED rows = 0.
- No PROD API/SQL or deployment change performed.
- J0 seed status remains `EN_COURS` but Control Tower's computed status should be `BLOQUE` while the C5 issue is open.
- No authorization given for an acceptance or J1 start.

## Next admission evidence needed

1. Run an authenticated **hosted Preview** HTTP test with a legitimate Vercel-authenticated operator session and OroTitan admin session. Use `vercel curl` or short-lived OIDC if already authorized and available. Avoid disabling protection or asking for long-lived secrets.
2. Confirm the exact application 401/403/400 responses with corresponding safety flags and unchanged event counts. Distinguish Vercel SSO 401 from OroTitan application 401.
3. Log the test evidence to C5, resolve `issue:j0-c5-preview-http-sso-20261031` only after independent readback.
4. Request distinct GO to formally accept J0 (C8). Do **not** auto-accept on test success.

Reference Preview: `https://orotitan-research-shadow-preview-45farwn7z-orotitan.vercel.app/admin/research-factory/control-tower`.
