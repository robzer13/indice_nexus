# Benchmark execution registry — manual rollback runbook

Scope: ONLY the noncanonical benchmark objects introduced by `migrations/20261004_orotitan_engine_benchmark_execution_registry.sql` at vnext commit `a10e2e61fa9815c865762e84437296f08c08b990`.

The forward migration is unchanged. Operator-only SQL lives in `ops/benchmark/rollback-execution-registry.sql` **outside automatic migrations**. Never schedule it or expose it via an API.

## Safety preconditions

1. Obtain a separate explicit GO identifying the target Supabase project and exact SQL. Independently verify a restorable backup or an adequate recovery strategy first.
2. Disable the benchmark runner and ensure no in-flight claim/write operations remain. **Any claim or execution row causes fail-closed refusal**, even when there is no completed execution; reservations must not be recycled.
3. Run using the owning database role. The script locks both tables with `ACCESS EXCLUSIVE NOWAIT`, verifies schema and ownership identity, table columns and exact default sets, the full ordered constraint definitions against independent transaction-local reference DDL derived from the pinned original migration, index definitions/validity, full trigger semantics (including WHEN), RLS, table/column ACLs and exact per-function privilege/grant-option boundaries, and exact source identity for all four PL/pgSQL functions.
4. Only `DROP ... RESTRICT` is used. Catalog-tracked dependencies (such as external views) trigger atomic refusal; a conservative `pg_proc.prosrc` scan additionally refuses recognizable external PL/pgSQL references. **Neither mechanism reliably detects dynamic SQL assembled at runtime, application-side SQL strings, or all external consumers.** Before executing any live rollback, independently audit functions, procedures, application code, schedules and other clients for untracked references. If this audit cannot establish their absence, **do not execute**. Any SQL error rolls back all statements in the transaction. The script never uses `CASCADE`, deletes data, or touches canonical research/dossier/run/artifact tables.
5. Record before/after catalog evidence and operator outcome. No automated execution against a shared Supabase project is permitted.

## Schema recovery is NOT migration-history recovery

This rollback **does not** edit `supabase_migrations.schema_migrations`. A successful schema rollback does not remove Supabase's recorded migration version; replay or deployment may subsequently consider the migration already applied. Inventory the precise migration record and reconcile history under a separately authorized, independently reviewed procedure. Never automatically delete/rewrite a migration-history record.

## Ephemeral PostgreSQL validation

VNext CI uses PostgreSQL 17, explicit local `PGHOST=127.0.0.1`, `PGPORT=5432`. The targeted script `tests/postgres/run-benchmark-rollback-tests.sh` creates only its own temporary database, runs original fixtures and original forward SQL, tests empty rollback and canonical row preservation, then confirms refusal with existing claims, execution rows, an external dependent view, an externally referencing PL/pgSQL routine, a count-preserving altered constraint, a substituted default, a changed index with the same name, an unexpected `service_role` MAINTAIN grant, a column-level grant, an internal-helper `EXECUTE` grant, an RPC `GRANT OPTION`, a trigger `WHEN` modification, altered table identity, and unauthorized role. Cleanup runs only after a positively successful `createdb`; a pre-existing name collision is never an authorization to drop the database.

Command under the isolated PostgreSQL service:

```bash
bash tests/postgres/run-benchmark-rollback-tests.sh
```

**Limits:** Successful CI is not proof of Supabase backup restorability, Point-in-Time Recovery availability, live service quiescence, exhaustive external dependency detection, or coherent migration-history rollback. Production/shared migration and any live rollback each require separate user GO and independent recovery preflight.
