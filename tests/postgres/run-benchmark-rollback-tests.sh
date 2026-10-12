#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
migration="$root/migrations/20261004_orotitan_engine_benchmark_execution_registry.sql"
rollback="$root/ops/benchmark/rollback-execution-registry.sql"
verify="$root/tests/postgres/benchmark-execution-registry-verify.sql"

# Prevent database creation/deletion through an accidental remote connection.
if [[ "$PGHOST" != "127.0.0.1" && "$PGHOST" != "localhost" ]]; then
 echo "ROLLBACK_TEST_REFUSED: PGHOST must explicitly be local" >&2; exit 1
fi
if [[ "$PGPORT" != "5432" ]]; then
 echo "ROLLBACK_TEST_REFUSED: PGPORT must explicitly be 5432" >&2; exit 1
fi
db="orotitan_benchmark_rollback_$$"
log="$(mktemp)"
db_created_by_test=0
cleanup() {
 # Never delete a database when creation did not positively succeed.
 if [[ "$db_created_by_test" == 1 ]]; then
  dropdb --if-exists "$db" >/dev/null 2>&1 || true
 fi
 rm -f "$log"
}
trap cleanup EXIT
new_case() {
 createdb "$db"
 db_created_by_test=1
 psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$fixture"
 psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$migration"
}
end_case() {
 if [[ "$db_created_by_test" != 1 ]]; then
  echo "ROLLBACK_TEST_REFUSED: no database ownership proof" >&2; exit 1
 fi
 dropdb "$db"
 db_created_by_test=0
}
assert_canonical() {
 psql -q -v ON_ERROR_STOP=1 -d "$db" -c "do \$check\$ begin
 if (select count(*) from public.companies)<>(select companies from public.fixture_legacy_before)
 or (select count(*) from public.snapshots)<>(select snapshots from public.fixture_legacy_before)
 or (select count(*) from public.market_prices)<>(select market_prices from public.fixture_legacy_before)
 or (select count(*) from public.market_sync_runs)<>(select market_sync_runs from public.fixture_legacy_before)
 or (select row_to_json(s)::text from public.snapshots s where id=1)<>(select snapshot_value from public.fixture_legacy_before)
 then raise exception 'CANONICAL_DATA_CHANGED'; end if; end \$check\$;"
}
assert_present() {
 psql -q -v ON_ERROR_STOP=1 -d "$db" -c "do \$check\$ begin
 if to_regclass('public.orotitan_engine_benchmark_executions') is null
 or to_regclass('public.orotitan_engine_benchmark_claims') is null
 or to_regprocedure('public.record_orotitan_benchmark_execution(jsonb)') is null
 or to_regprocedure('public.claim_orotitan_benchmark_execution(text,text,text,integer,text)') is null
 then raise exception 'FAILED_ROLLBACK_CHANGED_SCHEMA'; end if; end \$check\$;"
}
expect_reject() {
 local label="$1"
 if psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$rollback" >"$log" 2>&1; then
   echo "Unexpected rollback success: $label" >&2; exit 1
 fi
 if ! grep -Eq 'ROLLBACK_REFUSED_|cannot drop' "$log"; then
   echo "Unexpected rollback error for $label:" >&2; cat "$log" >&2; exit 1
 fi
 assert_present
 assert_canonical
}
# Empty benchmark: exact objects removed; canonical rows unchanged.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$rollback"
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "do \$check\$ begin
 if to_regclass('public.orotitan_engine_benchmark_executions') is not null
 or to_regclass('public.orotitan_engine_benchmark_claims') is not null
 or to_regprocedure('public.orotitan_benchmark_dimensions_valid(jsonb)') is not null
 or to_regprocedure('public.reject_orotitan_benchmark_execution_mutation()') is not null
 or to_regprocedure('public.record_orotitan_benchmark_execution(jsonb)') is not null
 or to_regprocedure('public.claim_orotitan_benchmark_execution(text,text,text,integer,text)') is not null
 then raise exception 'ROLLBACK_OBJECTS_REMAIN'; end if; end \$check\$;"
assert_canonical
end_case
# A claim without execution must still block loss of paid-inference identity.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "set role service_role; select public.claim_orotitan_benchmark_execution('X','Y','V2REF-010',1,repeat('1',64));"
expect_reject "nonempty claims"
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "do \$check\$ begin
 if (select count(*) from public.orotitan_engine_benchmark_claims)<>1 then raise exception 'CLAIM_LOST'; end if; end \$check\$;"
end_case
# Reuse existing validated benchmark script, which records an execution and claim.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$verify"
expect_reject "nonempty executions"
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "do \$check\$ begin
 if (select count(*) from public.orotitan_engine_benchmark_executions)<>1 then raise exception 'EXECUTION_LOST'; end if; end \$check\$;"
end_case
# External dependency must cause RESTRICT failure, atomic rollback of prior DROP.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "create view public.rollback_external_reference as select execution_id from public.orotitan_engine_benchmark_executions"
expect_reject "unexpected external dependency"
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "select 1 from public.rollback_external_reference limit 1"
end_case
# Static PL/pgSQL dependencies are not pg_depend-backed; reject explicitly.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
create function public.rollback_external_plpgsql_reader() returns bigint
language plpgsql as $fn$
declare v_count bigint;
begin
  select count(*) into v_count from public.orotitan_engine_benchmark_executions;
  return v_count;
end;
$fn$;
SQL
expect_reject "untracked external PL/pgSQL routine reference"
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "select public.rollback_external_plpgsql_reader()"
end_case
# Same constraint name/count/type, altered definition: must refuse.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "alter table public.orotitan_engine_benchmark_executions drop constraint orotitan_engine_benchmark_time_check; alter table public.orotitan_engine_benchmark_executions add constraint orotitan_engine_benchmark_time_check check (finished_at is not null)"
expect_reject "modified constraint definition with unchanged count"
end_case
# Missing default + replacement on another column retains total default count: reject.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "alter table public.orotitan_engine_benchmark_executions alter column execution_id drop default; alter table public.orotitan_engine_benchmark_executions alter column model_provider set default 'unexpected'"
expect_reject "default substitution preserving count"
end_case
# Same index name with changed key order must be rejected.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "drop index public.orotitan_engine_benchmark_campaign_case_idx; create index orotitan_engine_benchmark_campaign_case_idx on public.orotitan_engine_benchmark_executions (campaign_id, case_id, phase_id, repetition_index)"
expect_reject "changed index definition preserving name"
end_case
# Privilege widening must be rejected (PostgreSQL 17 MAINTAIN is not SELECT).
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "grant maintain on table public.orotitan_engine_benchmark_executions to service_role"
expect_reject "unexpected service_role MAINTAIN privilege"
end_case
# A table can acquire column-level grants with the table-level ACL unchanged.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "grant select (campaign_id) on table public.orotitan_engine_benchmark_executions to authenticated"
expect_reject "unexpected column-level authenticated SELECT"
end_case
# Internal helper functions must never become callable by service_role.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "grant execute on function public.orotitan_benchmark_dimensions_valid(jsonb) to service_role"
expect_reject "unexpected helper function EXECUTE"
end_case
# Even authorized RPCs must not offer regrant privileges.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "grant execute on function public.record_orotitan_benchmark_execution(jsonb) to service_role with grant option"
expect_reject "unexpected RPC GRANT OPTION"
end_case
# Preserve append-only trigger semantics, not just the trigger's name/function.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "drop trigger orotitan_engine_benchmark_executions_immutable on public.orotitan_engine_benchmark_executions; create trigger orotitan_engine_benchmark_executions_immutable before update or delete on public.orotitan_engine_benchmark_executions for each row when (false) execute function public.reject_orotitan_benchmark_execution_mutation()"
expect_reject "trigger WHEN(false) semantic drift"
end_case
# Inheritance/partition relationships are not harmless: refuse other-owned parents.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "create table public.rollback_external_parent (like public.orotitan_engine_benchmark_executions including defaults); alter table public.orotitan_engine_benchmark_executions inherit public.rollback_external_parent"
expect_reject "unexpected inherited parent relationship"
end_case
# A new identity column owns an unexpected sequence, even when type/nullability match.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "alter table public.orotitan_engine_benchmark_executions alter column repetition_index add generated by default as identity"
expect_reject "new generated identity sequence"
end_case
# An RPC return-contract mutation is not covered by regprocedure input signature.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
do $alter_rpc$
declare v_ddl text;
begin
 select pg_get_functiondef('public.claim_orotitan_benchmark_execution(text,text,text,integer,text)'::regprocedure) into v_ddl;
 execute 'drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict';
 v_ddl := regexp_replace(v_ddl, 'RETURNS[[:space:]]+jsonb', 'RETURNS text', 'i');
 if v_ddl !~* 'RETURNS[[:space:]]+text' then
   raise exception 'RETURN_TYPE_TEST_REWRITE_FAILED';
 end if;
 execute v_ddl;
 execute 'revoke all on function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) from public, anon, authenticated, service_role';
 execute 'grant execute on function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) to service_role';
end;
$alter_rpc$;
SQL
expect_reject "RPC return-type contract mutation"
end_case
# Schema drift must fail before any DROP.
new_case
psql -q -v ON_ERROR_STOP=1 -d "$db" -c "alter table public.orotitan_engine_benchmark_claims add column unexpected text"
expect_reject "schema identity mismatch"
end_case
# Privilege check: unauthorized operator cannot run the rollback.
new_case
if psql -q -v ON_ERROR_STOP=1 -d "$db" -c 'set role authenticated' -f "$rollback" >"$log" 2>&1; then
 echo "Unauthorized rollback succeeded" >&2; exit 1
fi
assert_present
assert_canonical
end_case
echo 'benchmark execution registry rollback: PASS'
