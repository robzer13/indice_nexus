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
cleanup() { dropdb --if-exists "$db" >/dev/null 2>&1 || true; rm -f "$log"; }
trap cleanup EXIT
new_case() {
 createdb "$db"
 psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$fixture"
 psql -q -v ON_ERROR_STOP=1 -d "$db" -f "$migration"
}
end_case() { dropdb "$db"; }
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
