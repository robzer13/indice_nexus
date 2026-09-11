# OroTitan Equity Research V1 — I0 Live Baseline / Safety Lock

## Repository baseline

```text
REPOSITORY
= robzer13/indice_nexus

BASELINE
= main@2abdb68ee7647d3d4873fd8b2ab77773fd011008

IMPLEMENTATION BRANCH
= feat/orotitan-equity-v1-i0-i1
```

No production migration has been applied by this branch.

## Canonical contract fingerprints

```text
04_SCREENER_SCHEMA_V1_PATCHED.json
SHA256
= bf407ca217553521586ba5f6002180ff6522700b4671986079ea6ed577604ede

04_INTEGRATION_SPEC_V1_PATCHED.md
SHA256
= f4d82ee65a9d653ebbb122d5fed04f90b722de8e0daa90844dc7fb1705ecf8b7

05_PHYSICAL_IMPLEMENTATION_INVENTORY_V1.md
SHA256
= 9b5143092d908a4daf81f761da2fb74fec96e3ae59df7efb9ca053003100f6d3
```

## Verified live Supabase target

```text
PROJECT_REF
= cugpgtzygqqlxetyeven

PROJECT_NAME
= orotitan-screener

REGION
= eu-west-3

STATUS
= ACTIVE_HEALTHY

POSTGRES_VERSION
= 17.6.1.166
```

No secret, service-role token, password, or connection string is stored here.

## Live schema fingerprint

Fingerprint basis:

```text
public tables
columns
constraints
indexes
views
triggers
public functions
RLS flags
policies
```

```text
LIVE_SCHEMA_SHA256
= 263623de6832f903668b42a7b3ded532b5a421a2511a7da535e9bacb3e9d856a
```

This fingerprint is a migration baseline only. Any later production migration must compare against a fresh live fingerprint before applying DDL.

## Verified live row counts

```text
companies
= 8

snapshots
= 8

market_prices
= 31

market_sync_runs
= 14
```

## Verified live structural facts

```text
public.companies
= PRESENT

public.snapshots
= PRESENT

public.market_prices
= PRESENT

public.market_sync_runs
= PRESENT

public.latest_company_state
= PRESENT
```

RLS:

```text
companies
= ENABLED

snapshots
= ENABLED

market_prices
= ENABLED

market_sync_runs
= ENABLED

RLS POLICIES
= NONE
```

Direct table privileges:

```text
anon
= NO SELECT / INSERT / UPDATE / DELETE

authenticated
= NO SELECT / INSERT / UPDATE / DELETE

service_role companies
= SELECT / INSERT / UPDATE, NO DELETE

service_role snapshots
= SELECT / INSERT, NO UPDATE / DELETE

service_role market_prices
= SELECT / INSERT, NO UPDATE / DELETE

service_role market_sync_runs
= SELECT / INSERT, NO UPDATE / DELETE
```

Expected immutable triggers are present and enabled:

```text
snapshots_immutable
= PRESENT

market_sync_runs_immutable
= PRESENT

companies_set_updated_at
= PRESENT
```

## Live / repository conclusion before I1

```text
REPOSITORY_LIVE_SCHEMA_MATCH
= YES

MATERIAL_SCHEMA_DRIFTS
= 0

KNOWN LEGACY IMPLEMENTATION_CONFLICTS
= 8

NEW_IMPLEMENTATION_CONFLICTS
= 0

READY_TO_IMPLEMENT
= YES
```

The eight known conflicts remain exactly those documented in
`05_PHYSICAL_IMPLEMENTATION_INVENTORY_V1.md`.

## Important live data observations

```text
legacy score_components
= HETEROGENEOUS / NOT CANONICALLY BACKFILLABLE

AUTO quote_unit
= MINOR

other current securities
= MAJOR

duplicate market_prices (company_id, as_of) groups
= 5
```

These facts require controlled migration logic later. They do not authorize rewriting legacy history.

## Security observations

Supabase security advisory currently reports:

```text
RLS enabled with no policy
= 4 tables / expected by current server-only design

mutable search_path
= 3 legacy trigger functions
```

I1 does not rewrite those legacy functions. New I1 trigger functions must use a fixed `search_path`.

## I0 safety invariants

Before any future production DDL:

1. Verify project ref remains `cugpgtzygqqlxetyeven`.
2. Recompute the live schema fingerprint.
3. Confirm row-count/profile drift is understood.
4. Confirm a recoverable backup/export exists.
5. Do not reinterpret or overwrite legacy snapshots.
6. Do not map legacy score fields into canonical OQS/OVS/Investment Score.
7. Do not map legacy OroTitan fields into the canonical terminal gate.
8. Do not rewrite market prices while identity migration is being established.

## I0 status

```text
I0_LIVE_BASELINE
= LOCKED

PRODUCTION_DATABASE_MODIFIED
= NO

LEGACY_DATA_MODIFIED
= NO

READY_FOR_I1_CODE
= YES
```
