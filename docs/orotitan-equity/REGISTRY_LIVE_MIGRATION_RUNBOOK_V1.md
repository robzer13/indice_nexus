# OroTitan Registry V1 — Live Migration Runbook

Status: PREPARED, NOT AUTHORIZED FOR EXECUTION
Date: 2026-09-14

This runbook prepares the production migration only. It does not authorize applying DDL, creating a live run, ingesting research artifacts, or publishing a canonical snapshot.

## 1. Frozen repository target

```text
repository = robzer13/indice_nexus
main_sha   = 146bc127c62aa867ae961bbf16c3b03b292d9571
```

Registry migrations are applied in this exact order and no other order:

```text
1. migrations/20260914_orotitan_registry_v1_1_core.sql
2. migrations/20260914_orotitan_registry_v1_2_guards_rls.sql
3. migrations/20260914_orotitan_registry_v1_3_rpcs.sql
4. migrations/20260914_orotitan_registry_v1_4_manifest_authority.sql
5. migrations/20260914_orotitan_registry_v1_5_contract_pin_guards.sql
```

The disposable PostgreSQL integration runner uses the same order and refuses non-local database hosts.

## 2. Verified production target

```text
project_ref  = cugpgtzygqqlxetyeven
project_name = orotitan-screener
region       = eu-west-3
status       = ACTIVE_HEALTHY
postgres     = 17.6.1.166 / engine 17
organization = OroTitan
plan         = free
```

Current production migration history before Registry V1:

```text
20260911123706 cloud_backup_pre_i1_20260911
20260911123940 orotitan_equity_i1_identity_20260911
20260911170015 orotitan_equity_i3_snapshot_persistence_20260911
20260912065624 orotitan_equity_i3b_validated_snapshot_writer_20260912
20260912070836 orotitan_equity_i4a_identity_completion_20260912
```

## 3. Read-only baseline observed during preparation

Observed on 2026-09-14 before any Registry production DDL:

```text
companies                    = 8
snapshots                    = 8
market_prices                = 31
market_sync_runs             = 14
issuers                      = 8
securities                   = 8
research_dossiers            = 8
legacy_company_identity_map  = 8
research_snapshots           = 0

legacy_rowset_md5   = 17bab890e9be8f52f803da3b133b03fa
identity_rowset_md5 = e22c8669935df331df2854f26f4290ba
```

Dependency/schema preparation facts:

```text
pgcrypto extensions.digest(bytea,text) = PRESENT
pgcrypto schema                         = extensions
Registry relations                     = ABSENT
Registry functions/RPCs                = ABSENT
research_snapshots_snapshot_dossier_key= PRESENT / UNIQUE
research_dossiers_current_snapshot_fkey= PRESENT
long transactions > 5 min              = 0 at preparation check
```

The `research_snapshots(snapshot_id, dossier_id)` FK target is backed by the unique index created by I3. A check limited to `pg_constraint` is insufficient because this target is intentionally implemented as a unique index.

## 4. Private artifact storage prerequisite

Verified live:

```text
orotitan-text-artifacts-v1  public=false objects=0
orotitan-source-files-v1    public=false objects=0
matching storage policies   = 0
```

No source document may be ingested until Registry migration is complete and a later run is explicitly authorized.

## 5. Backup gate — HARD BLOCK BEFORE DDL

The OroTitan Supabase organization is on the Free plan.

Current Supabase backup guidance states that automatic daily backups are provided for Pro, Team and Enterprise projects; Free projects should maintain their own logical exports with `supabase db dump` / `pg_dump`.

Therefore:

```text
BACKUP_GATE = NOT YET PROVEN
LIVE_DDL    = BLOCKED UNTIL BACKUP_GATE = PASS
```

Before the future `GO APPLY REGISTRY LIVE MIGRATION`, create and retain an off-site logical database dump from the same production state used by the final preflight. Record at minimum:

```text
backup timestamp
project ref
backup file SHA-256
preflight legacy_rowset_md5
preflight identity_rowset_md5
main SHA
```

Reference: https://supabase.com/docs/guides/platform/backups

A migration-history entry named `cloud_backup_pre_i1_20260911` is not, by itself, proof that a current recoverable pre-Registry backup exists.

## 6. Mandatory preflight immediately before DDL

Run:

```text
tests/postgres/registry-v1-live-readonly-preflight.sql
```

The script opens a read-only transaction and must return:

```text
database_gate_pass = true
registry_relations_absent = true
registry_functions_absent = true
pgcrypto_digest_present = true
snapshot_dossier_unique_index = true
current_snapshot_fk_present = true
private buckets = private + empty + no matching policies
long_transactions_over_5m = 0
```

Capture the returned `legacy_rowset_md5`, `identity_rowset_md5`, row counts and main SHA. If any value differs materially from this runbook, stop and reconcile drift before DDL.

## 7. Security exposure between V1.1 and V1.2

The production objects created by prior OroTitan migrations are owned by `postgres`.

Current `postgres` default table ACL in `public` does not grant SELECT / INSERT / UPDATE / DELETE to `anon` or `authenticated`. V1.2 then explicitly revokes all table rights from public client roles, enables RLS on all five Registry tables, and grants only SELECT to `service_role`; mutation is performed through controlled RPCs.

The five migrations must nevertheless be applied consecutively in one controlled maintenance sequence. No application or user workflow may use Registry objects until V1.5 and the postflight are complete.

## 8. Production application sequence — FUTURE AUTHORIZATION ONLY

On explicit future authorization only:

```text
PRECHECK
→ verify main SHA unchanged
→ verify exact project ref
→ verify BACKUP_GATE = PASS
→ run read-only preflight
→ capture hashes/counts

DDL
→ apply V1.1 core
→ STOP on any error
→ apply V1.2 guards/RLS
→ STOP on any error
→ apply V1.3 controlled RPCs
→ STOP on any error
→ apply V1.4 manifest authority supersession
→ STOP on any error
→ apply V1.5 13-pin completeness guard
→ STOP on any error

POSTCHECK
→ run read-only postflight
→ compare legacy/identity hashes with immediate preflight
→ verify Registry row counts remain exactly zero
→ run Supabase security advisor
→ run Supabase performance advisor
```

Each migration is transactional. A failure inside one migration must not be followed by the next migration.

If an earlier migration committed and a later migration fails, do not improvise destructive rollback. The Registry is expected to contain zero rows and remains outside the authorized runtime path; preserve the failed state, diagnose it, and execute only a separately reviewed corrective or teardown migration.

## 9. Mandatory postflight

Run:

```text
tests/postgres/registry-v1-live-readonly-postflight.sql
```

Required result:

```text
registry_tables                  = 5
all_registry_tables_rls          = true
registry_policies                = 0
direct_registry_write_grants     = 0
controlled_rpcs                  = 11
rpc_security_failures            = 0
rpc_privilege_failures           = 0
status_view_present              = true
active_manifest_deferred_fk      = true
artifact_manifest_deferred_fk    = true
contract_pin_completeness_check  = true
postflight_structure_pass        = true
```

All five Registry row counts must remain `0` immediately after migration.

The postflight legacy and identity row hashes must equal the immediately preceding preflight hashes. Registry DDL is additive and must not reinterpret or mutate pre-existing legacy/canonical rows.

Do NOT execute `tests/postgres/registry-v1-verify.sql` against production: its operational matrix intentionally creates test runs, events and artifact rows. It is restricted to disposable/local PostgreSQL.

## 10. Supabase 2026 compatibility check

Reviewed against the current Supabase changelog/docs during preparation.

Relevant platform change: new `public` tables are moving to explicit Data API exposure/grants. Registry V1 is compatible because it explicitly revokes direct client writes, enables RLS, and exposes controlled server RPCs rather than relying on automatic table exposure.

No Registry migration depends on deprecated extension-version pinning, Realtime schema mutation, GraphQL introspection, or the removed analytics endpoint.

## 11. Boundary after successful migration

Even after all five migrations and postflight pass:

```text
Registry schema installed = YES
Registry rows created      = NO
Research run authorized    = NO
Artifact ingestion         = NO
Canonical publication      = NO
```

The next operational boundary must be separate. A Registry schema deployment is not authorization for a first live OroTitan run.

## 12. Preparation status

```text
REPOSITORY HEAD LOCK          = PASS
LIVE PROJECT IDENTITY         = PASS
CANONICAL DEPENDENCIES        = PASS
PGCRYPTO DEPENDENCY           = PASS
REGISTRY NAME COLLISIONS      = PASS
PRIVATE STORAGE               = PASS
READ-ONLY PREFLIGHT SCRIPT    = PREPARED
READ-ONLY POSTFLIGHT SCRIPT   = PREPARED
LOCAL INTEGRATION MATRIX      = PASS FROM PR #17
BACKUP GATE                   = BLOCKED / REQUIRES CURRENT LOGICAL EXPORT
PRODUCTION DDL                = NOT AUTHORIZED
```
