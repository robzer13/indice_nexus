# OroTitan V2 Production Activation Record

Date: 2026-09-15  
Status: FINAL  
Scope: OroTitan Equity Research V2 production admission

## 1. Authorization

Explicit activation authorization received:

```text
GO ACTIVATE OROTITAN V2
```

This authorization covered V2 production activation only. It did not authorize a company research run and did not authorize canonical publication.

## 2. Code authority

Repository:

```text
robzer13/indice_nexus
```

Activation implementation PR:

```text
#23 feat(equity): activate OroTitan V2 runtime and canonical product
```

Immutable V2 authority source commit:

```text
86b227a75275cf4aaec6ef61ea2a27e87e2bfec7
```

Activation merge commit on `main`:

```text
7ee7e8efef2737d0f929bc3be232fac162b6f60b
```

The merge commit is GitHub signature-verified and preserves the source commit in history.

## 3. Contract Pin Pack V2

Manifest:

```text
contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json
```

Logical pins:

```text
13
```

Contract-set SHA-256:

```text
1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

Production Registry verification before activation:

```text
V2_PINS_COMPLETE = TRUE
DB_CONTRACT_SET_SHA256 = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
CONTRACT_SET_MATCHES = TRUE
CONTROLLED_RPC_COUNT = 11
REGISTRY_DIRECT_WRITE_GRANTS = 0
```

## 4. CI and deployment admission

V2 source/hash CI:

```text
run 34952398657 / Screener CI #141 / PASS
```

Final branch CI with Contract Pin Pack V2:

```text
run 34952725886 / Screener CI #142 / PASS
```

PR CI:

```text
Screener CI #143 / PASS
```

Post-merge `main` CI:

```text
run 34953042242 / Screener CI #144 / PASS
```

Validated surfaces included:

```text
lint
strict TypeScript typecheck
239+ automated tests including V2-01 through V2-45
all PostgreSQL migration matrices
Registry V1 regression matrix
V2 canonical writer matrix
Next.js production build
```

Vercel PR preview and post-merge production deployment both reported success before database activation was declared complete.

## 5. Production database migration

Migration source:

```text
migrations/20260915_orotitan_v2_canonical_snapshot_writer.sql
```

Supabase migration record:

```text
name    = orotitan_v2_canonical_snapshot_writer_20260915
version = 20260915093437
```

Migration result:

```text
SUCCESS
```

The migration is additive. It preserves the V1 writer and existing V1 snapshots and adds a dedicated V2 writer plus V1/V2 contract-schema firewalls.

## 6. Data-preservation postflight

Canonical snapshot state before and after migration:

```text
row_count  = 1
rowset_md5 = b81d57cf3c71329ae1b8294c4fd058af
preserved  = TRUE
```

Research dossier state before and after migration:

```text
row_count  = 8
rowset_md5 = 5cb202f29535a1f68c75b05b94669003
preserved  = TRUE
```

Registry row counts before and after migration:

```text
runs      = 2
stages    = 5
artifacts = 66
edges     = 39
events    = 21
preserved = TRUE
```

Qualys canonical state:

```text
current snapshot preserved = TRUE
contract_version            = 04_SCREENER_SCHEMA_V1
schema_version              = 1.0.0
```

No V2 canonical snapshot was created by activation:

```text
V2 snapshot rows      = 0
rows with v2_product  = 0
```

## 7. Persistence and privilege postflight

```text
V1 writer exists                 = TRUE
V2 writer exists                 = TRUE
V1 SECURITY DEFINER              = TRUE
V2 SECURITY DEFINER              = TRUE
V1 search_path locked            = TRUE
V2 search_path locked            = TRUE
V1 PUBLIC/anon/authenticated     = NO EXECUTE
V2 PUBLIC/anon/authenticated     = NO EXECUTE
V1 service_role                  = EXECUTE
V2 service_role                  = EXECUTE
service_role direct snapshot DML = NONE
```

Active snapshot constraints now enforce:

```text
V1 contract + V1 schema only
OR
V2 contract + V2 schema only

V1 payloads cannot contain v2_product.
V2 payloads must contain an object v2_product.
```

## 8. Storage and security postflight

Private artifact stores remained private and empty during activation.

Security advisor findings after migration were unchanged from the pre-activation baseline:

```text
14 INFO: RLS enabled with no policy, intentional existing access model
3 WARN: legacy mutable search_path functions, pre-existing and outside V2 activation scope
0 new V2 security warning
```

## 9. Final production state

```text
V2_DESIGN_STATUS     = FROZEN
V2_PRODUCTION_STATUS = ACTIVE_FOR_NEW_RUNS
V1_EXISTING_RUNS     = GRANDFATHERED
```

Operational consequences:

```text
NEW AUTHORIZED RUNS
-> use the V2 Contract Pin Pack and V2 orchestration

EXISTING V1 RUNS / SNAPSHOTS
-> remain immutable historical truth

QUALYS CURRENT V1 SNAPSHOT
-> remains canonical until a future authorized refresh

CANONICAL PUBLICATION
-> still requires a separate GO PUBLISH <COMPANY>
```

No company run and no canonical publication occurred as part of this activation.
