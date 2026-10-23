# OroTitan — Product MVP 02 / Shared Research inbox

**Release status:** REVIEW ONLY, GitHub DRAFT stacked on Product MVP 01 (#419).
**Product criterion:** two independent devices using **one authoritative shared
database** retrieve the same durable request ID for the same active company.
This is an **intake request**, not the start of an OroTitan Research run.

## Actual changes

- SQL proposal `docs/orotitan-equity/proposals/PRODUCT_SHARED_RESEARCH_INBOX_PG17_REVIEW_ONLY.sql`
  defines a single immutable Research intake table keyed to an existing
  Screener company. UUID and company metadata are generated/copied inside
  PostgreSQL, not trusted from a browser or model. An active-company check is
  enforced in the INSERT transaction. Unique `company_id` prevents
  duplicate request rows even during simultaneous submissions.
- Only backend `service_role` may SELECT/INSERT. RLS is enabled, and
  `anon`/`authenticated` are expressly denied direct table access.
  Neither UPDATE nor DELETE is granted. Ingestion **cannot create** a
  canonical issuer, security, dossier, Registry run, or analysis.
- Candidate Supabase adapter uses the current application SDK on the
  server to insert **only company_id** and returns DB-generated receipts.
  Only `23505` can be treated as a duplicate, and the original row must
  be visible before claiming idempotent success.
- The existing `/admin/research-requests` UI is reused, not duplicated.
  In `NODE_ENV=development`, it can use local disk **or** an isolated
  Supabase **test** project with the reviewed SQL table present.
  The shared-test mode is disabled by default. Its URL must be HTTPS with
  the exact explicitly allowlisted nonproduction project reference; the
  production OroTitan project reference is hard-denied. The secret key is
  server-only. The selected company's details are read from the same test
  project's Screener table, not from a second unrelated source.
- Existing login guard applies to the admin page and server action.

## Acceptance tests

| Scenario | Expected |
|---|---|
| Local mode on development laptop | Existing #419 UX and file persistence |
| Shared mode on isolated test project | Same URL, separate backing store |
| Two independent PostgreSQL sessions | Both read identical request UUID |
| Concurrent duplicate requests | One insert, one existing receipt; one row |
| Unknown/inactive company | Rejected at database layer |
| Malicious company name, ticker or UUID | DB supplies authoritative metadata and UUID |
| `anon`/`authenticated` role access | No SELECT/INSERT permissions |
| Request creation | Only `REQUESTED_NOT_EXECUTED` and `registry_run_id=NULL` |
| Existing Screener/Registry records | No generated analysis, no Registry mutation |

The CI constructs an **ephemeral PostgreSQL 17** database using existing
Screener `schema.sql`, applies the proposal SQL inside the fixture only,
and runs authorization, metadata, immutability and two-real-connection
race tests. Adapter unit tests verify server SDK behavior.

## Known boundaries

This is **not an actual live multi-device deployment**. No remote test
project has been provisioned or migrated by this PR. The database contract
and SQL concurrency tests prove the shared-storage behavior in an isolated
PostgreSQL service; they do not prove real HTTP/browser/device syncing.

The app currently uses **one shared admin password**, not distinct user
accounts. This is a single-operator workspace MVP. It does not implement
tenant isolation, bearer/mobile OAuth, independent issuer/ISIN provenance,
contract-pack/cutoff-scoped renewal, a Research worker, or Registry entry.
The partial D30–D38 prototypes must be selectively reconciled, not
automatically merged. The one-request-per-company rule is intentionally
simple for the MVP and must be upgraded for repeated analyses.

Local/test access is disabled unconditionally when `NODE_ENV=production`.
The test service key must **never** be the live OroTitan production key.
No Supabase Free production migration, Vercel deployment, OAuth write scope,
purchase, paid project, merge, Research start, Besi evidence admission or
Deep Dive change is approved.

## Precise next go/no-go product decision

**Next milestone after CI:** manually test this web flow against a separately
approved, nonproduction shared instance from two real browsers. Then seek
explicit approval for the minimum audited schema + backend release to
OroTitan's existing Free project. No write rights or production migrations
should be enabled merely because this CI passes.

Once that shared persistent inbox is proved and separately authorized,
connect ChatGPT commands to **the same backend**, not another shadow queue.
