# OroTitan Run Factory — P0-C Canonical Identity Onboarding Candidate V1

Date: 2026-10-31
Status: ISOLATED SHADOW SOFTWARE CANDIDATE; NO REGISTRY MUTATION
Scope: onboarding/identity discovery and reviewed plan, not a Research run.

## Why this exists

Historical blocker: `companies` is a legacy Screener catalog, while canonical
Research requires `issuers` + `securities` + `research_dossiers` and an
optional `legacy_company_identity_map` reference. Populating the legacy
catalog alone does **not** authorize an Equity Research run.

The SHADOW schema inspected on 2026-10-31 has:
- unique index `lower(exchange),lower(ticker)`;
- unique partial index for one primary security per issuer;
- unique partial index for one active dossier per issuer;
- **no globally unique ISIN index**;
- no known approved atomic canonical identity onboarding RPC.
- separate P0-B read-only run preflight and existing
  `create_orotitan_run` / `bind_orotitan_run_identity` run RPCs.

**Important:** do NOT simply add a global unique constraint on ISIN:
multiple market listings may legitimately share one ISIN. An eventual
transaction must lock/validate the exact listing and issuer, detect
legitimate cross-listings, and prevent two concurrent creation attempts
from allocating duplicate canonical entities.

## Delivered in this branch

- Pure deterministic planner:
  `lib/orotitan-equity/product/runfactory-canonical-onboarding-candidate-v1.ts`.
- Admin-authenticated read-only SHADOW endpoint:
  `POST /admin/research-factory/onboarding/inspect`.
- Operator view:
  `/admin/research-factory/onboarding`.
- Reusable tests:
  `tests/runfactory-canonical-onboarding-candidate-v1.test.ts`.
- CI typecheck, lint, tests and production-compatible build.

Operator supplies one-time **ISIN + ticker + exchange**. The endpoint:
1. validates ISIN with ISO 6166/Luhn;
2. performs exact ISIN and listing discovery among canonical securities;
3. looks up an exact legacy Screener company by ticker/exchange;
4. retrieves any crosswalk and canonical issuer/dossier connections;
5. detects duplicate listing ISIN conflicts, same-ISIN cross-listings,
   name collisions, legacy duplication, orphaned/crosslinked records;
6. produces a candidate JSON with deterministic diagnostic SHA-256,
   explicit blockers, provenance type and read duration;
7. NEVER generates canonical UUIDs, claims issuer legal verification,
   writes to Supabase or invokes any stored procedure.

Existing canonical identity: candidate for reuse, no new entity.
Legacy-only record: unverified metadata suggestion, not legal proof.
Unknown name/issuer: controlled exception; never fabricated.
Source URL: optional pointer to be fetched and authenticated later,
not evidence by itself.

## Real L'Oreal SHADOW observation

ISIN `FR0000120321`; ticker `OR`; market `XPAR`.
SHADOW has zero matching canonical security and issuer records at the
time of the preflight. This is a SHADOW-only fact, not an assertion about
production or the company in the real world.
If no exact legacy Screener row exists, issuer name, country, trading
currency and legal linkage remain UNKNOWN and must not be inferred
from the ISIN prefix or ticker.

## Necessary next step for **actual** onboarding (not delivered)

A separately reviewed atomic, idempotent server-side workflow and a
controlled database RPC, with:
- exact authorized environment + explicit operator approval/GO;
- verified official issuer and listing document bytes, provenance,
  source date, independent ISIN-ticker-exchange cross-check;
- explicit legal entity, issuer vs security, financial reporting currency
  vs trading currency, primary listing, quote unit and price multiplier;
- transaction-scoped lock on normalized listing and ISIN namespace;
- exhaustive query for existing issuer, security, legacy crosswalk,
  active dossier and candidate episode before allocating UUIDs;
- no global UNIQUE(ISIN) assumption; preserve legitimate cross-listings;
- double-submit and concurrent retry idempotence, with read-after-write;
- privileges restricted to approved server role, no public execute;
- append-only auditable receipt and exact source record references;
- write issuer/security/dossier/map only in one transaction after
  authority gates PASS, with explicit rollback on any mismatch;
- no Research run created in this onboarding step and no publication;
- isolated PostgreSQL integration tests, concurrency tests and
  PR approval before any live DDL.

**Pre-admission invariant:** every candidate response has
`registryMutated=false`, `rpcExecuted=false`,
`issuerIdentityAdmitted=false`, `runCreated=false`,
`operationAuthorized=false`.

## Reuse metric

One verified issuer/security/dossier is reused across future cutoffs.
A new run should only require (1) identity lookup, (2) approved
versioned Contract Set, (3) cutoff and GO, rather than re-onboarding
the company.
