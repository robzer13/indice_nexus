#!/usr/bin/env bash
# D29b: review-only isolated PostgreSQL 17. Never reach an external Supabase DB.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo "D29b: refusing non-loopback PGHOST" >&2; exit 2 ;;
esac
if [[ "$PGUSER" != postgres || "$PGPORT" != 5432 ]]; then
  echo "D29b: refusing non-fixture PostgreSQL user/port" >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d29b_research_$$"
created=0
provisioned=0
cleanup() {
  if [[ "$created" == 1 ]]; then
    dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
  fi
  if [[ "$provisioned" == 1 ]]; then
    psql -X -v ON_ERROR_STOP=1 -d postgres -c       'DROP ROLE IF EXISTS orotitan_research_intake_writer,orotitan_research_intake_reader,orotitan_research_intake_receipt_executor,orotitan_research_intake_verifier_executor;'       >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
# Explicitly refuse a cluster with already-provisioned D29b roles: never
# tamper with a real private deployment or another test's global roles.
count="$(psql -XAt -d postgres -c "SELECT count(*) FROM pg_roles
WHERE rolname IN ('orotitan_research_intake_writer',
'orotitan_research_intake_reader','orotitan_research_intake_receipt_executor','orotitan_research_intake_verifier_executor')")"
if [[ "$count" != 0 ]]; then
  echo "D29b: private roles already exist; aborting" >&2; exit 2
fi
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
if (( version < 170000 || version >= 180000 )); then
  echo "D29b: PostgreSQL 17 required" >&2; exit 3
fi
psql -X -v ON_ERROR_STOP=1 -d "$db"   -f "$root/tests/postgres/d29b-research-private-fixture.sql" >/dev/null
# The SQL proposal creates ONLY roles and functions in the disposable DB.
psql -X -v ON_ERROR_STOP=1 -d "$db"   -f "$root/docs/orotitan-equity/proposals/D29B_RESEARCH_PRIVATE_RECEIPT_SQL_REVIEW_ONLY.sql"   >/dev/null
provisioned=1
psql -X -v ON_ERROR_STOP=1 -d "$db"   -f "$root/tests/postgres/d29b-research-private-verify.sql" >/dev/null
bash "$root/tests/postgres/d29b-research-private-race-verify.sh" "$db"
echo 'D29b Research private SQL scoped/negative PG17 tests PASS; production untouched.'
