#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;; esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
i1="$root/migrations/20260909_orotitan_equity_i1_identity.sql"
i3="$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"
verify="$root/tests/postgres/i3-verify.sql"
prefix="orotitan_i3_${$}"
databases=()
cleanup() { for database in "${databases[@]}"; do dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT

new_database() {
  db="${prefix}_$1"
  createdb "$db"
  databases+=("$db")
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$fixture" >/dev/null
}
apply() { psql -X -v ON_ERROR_STOP=1 -d "$1" -f "$2" >/dev/null; }

new_database main
apply "$db" "$i1"

before_legacy="$(psql -XAt -d "$db" -c "select (select count(*) from companies)||':'||(select count(*) from snapshots)||':'||(select count(*) from market_prices)||':'||(select count(*) from market_sync_runs)||':'||(select count(*) from issuers)||':'||(select count(*) from securities)||':'||(select count(*) from research_dossiers)||':'||(select count(*) from legacy_company_identity_map)")"
before_values="$(psql -XAt -d "$db" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"

apply "$db" "$i3"
apply "$db" "$i3"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$verify"

after_legacy="$(psql -XAt -d "$db" -c "select (select count(*) from companies)||':'||(select count(*) from snapshots)||':'||(select count(*) from market_prices)||':'||(select count(*) from market_sync_runs)||':'||(select count(*) from issuers)||':'||(select count(*) from securities)||':'||(select count(*) from research_dossiers)||':'||(select count(*) from legacy_company_identity_map)")"
after_values="$(psql -XAt -d "$db" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$before_legacy" = "$after_legacy"
test "$before_values" = "$after_values"
echo 'LEGACY_COUNTS_PRESERVED LEGACY_VALUES_PRESERVED I1_IDENTITY_COUNTS_PRESERVED'

# T26A: incompatible pre-existing shape remains intact after fail-closed rejection.
new_database incompatible
apply "$db" "$i1"
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create table public.research_snapshots (snapshot_id integer primary key);' >/dev/null
if psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$i3" >/dev/null 2>&1; then
  echo 'Expected incompatible research_snapshots migration failure' >&2
  exit 1
fi
test "$(psql -XAt -d "$db" -c "select to_regclass('public.research_snapshots')")" = 'research_snapshots'
test "$(psql -XAt -d "$db" -c "select format_type(atttypid, atttypmod) from pg_attribute where attrelid = 'public.research_snapshots'::regclass and attname = 'snapshot_id'")" = 'integer'
test "$(psql -XAt -d "$db" -c "select count(*) from pg_attribute where attrelid = 'public.research_snapshots'::regclass and attnum > 0 and not attisdropped")" = '1'
if psql -XAt -d "$db" -c "select count(*) from pg_constraint where conrelid = 'public.research_dossiers'::regclass and conname = 'research_dossiers_current_snapshot_fkey'" | grep -vq '^0$'; then
  echo 'Pre-existing shape test left pointer FK' >&2
  exit 1
fi

# T26B: a late collision after table creation rolls back every I3-A object.
new_database atomic
apply "$db" "$i1"
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create table public.i3_collision_fixture (x integer, y integer); create unique index research_snapshots_snapshot_dossier_key on public.i3_collision_fixture (x, y);' >/dev/null
if apply "$db" "$i3"; then
  echo 'Expected late index collision validation failure' >&2
  exit 1
fi
test "$(psql -XAt -d "$db" -c "select to_regclass('public.research_snapshots')")" = ''
test "$(psql -XAt -d "$db" -c "select to_regclass('public.i3_collision_fixture')")" = 'i3_collision_fixture'
test "$(psql -XAt -d "$db" -c "select to_regclass('public.research_snapshots_snapshot_dossier_key')")" = 'research_snapshots_snapshot_dossier_key'
test "$(psql -XAt -d "$db" -c "select count(*) from pg_constraint where conname in ('research_dossiers_dossier_issuer_key', 'securities_security_issuer_key', 'research_dossiers_current_snapshot_fkey')")" = '0'
test "$(psql -XAt -d "$db" -c "select count(*) from pg_proc where oid = to_regprocedure('public.prevent_orotitan_research_snapshot_mutation()')")" = '0'
echo 'T26 incompatible-shape and atomic-rollback regressions passed'