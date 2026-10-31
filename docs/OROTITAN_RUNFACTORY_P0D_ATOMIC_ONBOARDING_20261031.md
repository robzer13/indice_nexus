# OroTitan Run Factory P0-D — ATOMIC CANONICAL IDENTITY ONBOARDING
Date: 2026-10-31
Status: SHADOW candidate; production strictly out of scope.

## What this delivers (not just a dry run)

- A real PostgreSQL transaction in
  `public.orotitan_onboard_identity_shadow_v1` that can INSERT
  `issuers`, `securities`, `research_dossiers`, and optionally
  `legacy_company_identity_map`.
- Deterministic request-key and canonical payload SHA-256; exact retry
  returns stable IDs, different payload under same key is denied.
- Locks over the ISIN namespace and exchange/ticker namespace to prevent
  racing requests from creating duplicate canonical identities.
- Supports legitimate cross-listing of one ISIN under one existing issuer.
  A global UNIQUE(ISIN) is **intentionally absent**.
- Refuses same exchange/ticker with another ISIN, multiple issuers
  with the same ISIN, unexplained matching names, secondary primary listing,
  inconsistent currencies/unit/precision, legacy map collisions and
  invalid ISO-6166 check digits.
- One active dossier (new or reused); no snapshot and no Research run.
- A protected receipt table `orotitan_identity_onboarding_receipts_v1`
  with RLS and privileges restricted to service_role.
- Admin-authenticated, exact hosted-SHADOW-only POST endpoint with
  CSRF origin check, capped payload and independent readback of issuer,
  security, active dossier and idempotency receipt.
- Two-stage operator page:
  `/admin/research-factory/onboarding/execute`.
  P0-C dry-run first; reviewed primary references and exact
  `ONBOARD <ISIN>` confirmation second; transaction only on click.
- Role privilege: SQL function SECURITY INVOKER; PUBLIC/anon/authenticated
  do NOT have execute rights. The service key is never sent to browser.

## Critical provenance distinction

P0-D is **operational canonical identity onboarding after manual review**.
An HTTPS issuer URL and an HTTPS exchange URL are recorded and the operator
attests they independently reviewed the issuer, ISIN, market, ticker,
currency and quotation unit. P0-D does **not** download, archive, hash or
authenticate the linked source bytes automatically. Its receipt field
`source_automatically_verified=false` is not cosmetic: it must remain false.

The current admin authentication is a shared session rather than named
operator identity; audit can prove a valid admin session attested, not
which human person performed the review. This is an explicit audit gap
to resolve before production.

## Isolated PostgreSQL acceptance

Disposable Postgres 17+ and the original I1 schema:
`bash tests/postgres/run-orotitan-runfactory-p0d-tests.sh`.
The runner refuses all non-loopback hosts.

- 15 transaction tests: atomic first creation, exact replay, new-key reuse,
  mismatch key denial, listing/ISIN conflict, legitimate second listing,
  primary listing collision, currency conflict, independent checksum,
  missing attestation, missing source, no unintended mutations,
  public/anon/authenticated prohibition + receipt RLS, rollback after an
  invalid legacy link, and successful new legacy crosswalk.
- A second migration application is idempotent and preserves receipts.
- Two concurrent calls with different keys and same ISIN/listing resolve
  exactly one issuer, one security and one active dossier.
- Two concurrent requests with incompatible ISIN at the same listing
  result in exactly one success, one refusal, no duplicates.

P0-A / P0-B / P0-C and P0-D Node unit tests remain in `npm test`.

## Deployment discipline

1. Check Vercel project ID and deployment target are exact SHADOW Preview;
   inspect `isHostedShadowResearchEnabled` and connection env.
2. Validate isolated Postgres runner and application typecheck/lint/tests/build.
3. Apply SQL migration to the **SHADOW project reference only** through
   the Supabase migration tool and inspect RLS/grants/functions.
4. Test live SHADOW RPC inside SQL BEGIN/ROLLBACK, then verify row counters.
   No synthetic fixture should persist in SHADOW.
5. Open the admin page to perform a real manual-reviewed onboarding.
   The user must affirm the two independent source URLs and metadata.
6. Independent receipt readback confirms persisting issuer/security/dossier.
   Same form re-submitted with same reviewedAt must return REPLAYED.
7. Only THEN use P0-B to prepare a run; P0-D never calls run RPC.

## What is deliberately still not automatic

- Trusted publisher source capture / immutable byte receipts;
- autonomous legal-entity/ISIN cross-check from multiple external providers;
- named human approval log and role separation;
- run creation, stage admission, Research claim validation or publication;
- migration to or modification of production.

These are not hidden tasks. P0-D makes the identity creation transactional
and repeatable and confines necessary human judgment to a first-time review.
Subsequent runs reuse the same issuer and dossier.
