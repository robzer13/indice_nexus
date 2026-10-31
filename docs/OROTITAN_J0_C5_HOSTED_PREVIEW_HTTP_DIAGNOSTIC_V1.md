# OroTitan J0/C5 — Hosted Preview negative HTTP diagnostic

2026-10-31 · SHADOW only · No acceptance authority.

## Purpose and strict boundary
The existing J0 acceptance gate C5 is partially blocked by Vercel SSO:
an unauthenticated external curl receives a Vercel edge 401 before OroTitan
application code runs. This is NOT evidence of the application's own 401/403.

This diagnostic uses the **already logged-in Vercel/OroTitan administrator's
Preview request**. Only after verifying the active admin session does it
perform four actual HTTPS requests to the same exact Preview hostname,
forwarding the request's cookies **only to that same origin**. Neither cookies
nor tokens are logged, persisted in the journal or sent in the response.
The hostname is pinned to `VERCEL_URL` and must match the target project;
it cannot be supplied through the request body or query string.

No Deployment Protection settings, password, SSO or production environment
is altered. No external host can be selected as a fetch destination.

## Exact test cases

1. `CROSS_ORIGIN`: invalid `Origin` => **403 ORIGIN_MISMATCH**
2. `UNSUPPORTED_CONTENT_TYPE`: `text/plain` => **400 REQUEST_INVALID**
3. `UNEXPECTED_FIELDS`: disallowed JSON key => **400 J0_UNEXPECTED_FIELDS**
4. `FORGED_ACCEPTANCE`: `MILESTONE_ACCEPTED` input =>
   **400 J0_INPUT_INVALID**

All four expected errors must be generated as OroTitan JSON with
`accepted=false`, `stageStarted=false`, and a `no-store` response.
A Vercel edge HTML/401 response is rejected as NON-VERIFIED.

The payloads are deliberately non-mutating **even if a guard regresses**.
Before and after, the service checks the J0 event count, plus L’Oréal's run
status, version, stages, run artifacts and dossier snapshots. Differences
block the success receipt.

## Reproducible operator procedure

1. Open the exact new SHADOW Preview deployment URL, log into Vercel if
   prompted, and log into the OroTitan admin area.
2. Open `/admin/research-factory/control-tower`.
3. In **ACTION ADMINISTRATEUR — CONTRÔLE C5**, click
   **EXÉCUTER LE CONTRÔLE HTTP C5**.
4. The API issues the four negative HTTP requests over the protected
   Preview's network path and checks the Registry has not changed.
5. Only if all pass, it writes **one append-only J0 evidence event**
   with a deterministic key derived from the Git commit, then performs
   an independent stored-event readback.
6. Give the resulting event UUID to the reviewer. The reviewer separately
   re-reads it in SHADOW, checks the active issue and original run counts,
   and ONLY THEN may close
   `issue:j0-c5-preview-http-sso-20261031`.

Repeated calls use the same idempotency key. A conflicting replay fails.
The recorded proof is labelled `REPORTED` because the J0 SQL Registry's
automatic `VERIFIED_DIRECT_REGISTRY_READBACK` status is reserved for
`registry:run:<UUID>` evidence. Separate reviewer readback is mandatory.

No RPC to start Research, create stages/artifacts/snapshots, publish results,
accept J0 or advance J1 is reachable through this diagnostic.

## Limitations
This test proves actual application-generated HTTP 403 and 400 responses on
the hosted Preview using an authenticated administrator session. It does
not impersonate an unauthenticated browser inside Vercel SSO, and it does
not prove the application's own network 401 behind SSO. The separate
disposable local application HTTP 401 plus SHADOW anon/authenticated SQL
negative role tests remain evidence for that authorization boundary.
If Vercel SSO does not accept the server-forwarded same-origin cookies,
the test fails closed and C5 stays blocked. Do not relax SSO to make it pass.
