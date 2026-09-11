#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD

case "$PGHOST" in
  127.0.0.1|localhost|::1) ;;
  *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
migration="$root/migrations/20260909_orotitan_equity_i1_identity.sql"
verify="$root/tests/postgres/i1-verify.sql"
i3_migration="$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"
i3_verify="$root/tests/postgres/i3-verify.sql"
prefix="orotitan_i1_${$}"
databases=()
cleanup() {
  for database in "${databases[@]}"; do dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT

new_database() {
  db="${prefix}_$1"
  createdb "$db"
  databases+=("$db")
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$fixture" >/dev/null
}
apply() { psql -X -v ON_ERROR_STOP=1 -d "$1" -f "$migration" >/dev/null; }
expect_failure() {
  if apply "$1" >"/tmp/${1}.out" 2>&1; then
    echo "Expected migration failure in $1" >&2
    return 1
  fi
}

# T1/T2/T3 and T8-T14.
new_database main; main_db="$db"
apply "$main_db"
before="$(psql -XAt -d "$main_db" -c "select count(*)||':'||(select count(*) from securities)||':'||(select count(*) from research_dossiers)||':'||(select count(*) from legacy_company_identity_map) from issuers")"
ids_before="$(psql -XAt -d "$main_db" -c "select string_agg(issuer_id||':'||security_id||':'||dossier_id, ',' order by legacy_company_id) from legacy_company_identity_map")"
apply "$main_db"
after="$(psql -XAt -d "$main_db" -c "select count(*)||':'||(select count(*) from securities)||':'||(select count(*) from research_dossiers)||':'||(select count(*) from legacy_company_identity_map) from issuers")"
ids_after="$(psql -XAt -d "$main_db" -c "select string_agg(issuer_id||':'||security_id||':'||dossier_id, ',' order by legacy_company_id) from legacy_company_identity_map")"
test "$before" = "$after" && test "$ids_before" = "$ids_after"
psql -X -v ON_ERROR_STOP=1 -d "$main_db" -f "$verify" >/dev/null

# T4: incompatible existing table definition.
new_database partial
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create table public.issuers (issuer_id uuid primary key);' >/dev/null
expect_failure "$db"

# T5: expected security ID attached to the wrong issuer.
new_database security; apply "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -c "update securities set issuer_id='00000000-0000-4000-8000-000000000002' where issuer_id='00000000-0000-4000-8000-000000000001';" >/dev/null
expect_failure "$db"

# T6: expected dossier ID with the wrong candidate episode.
new_database dossier; apply "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -c "update research_dossiers set candidate_episode='40000000-0000-4000-8000-000000000001' where issuer_id='00000000-0000-4000-8000-000000000001';" >/dev/null
expect_failure "$db"

# T7: expected crosswalk attached to the wrong issuer.
new_database crosswalk; apply "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -c "update legacy_company_identity_map set issuer_id='00000000-0000-4000-8000-000000000002' where legacy_company_id='00000000-0000-4000-8000-000000000001';" >/dev/null
expect_failure "$db"

echo 'I1 PostgreSQL integration matrix T1-T14 passed'

# I3-A runs from the same legacy fixture after I1 has established canonical identity.
new_database i3
apply "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$i3_migration" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$i3_migration" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$i3_verify"

# T26: an incompatible pre-existing canonical table must fail closed.
new_database incompatible
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create table public.research_snapshots (snapshot_id integer primary key);' >/dev/null
if psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$i3_migration" >/dev/null 2>&1; then
  echo 'Expected incompatible research_snapshots migration failure' >&2
  exit 1
fi

echo 'I3-A PostgreSQL integration matrix T1-T26 passed'
