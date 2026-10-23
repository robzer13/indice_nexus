#!/usr/bin/env bash
# D30: only disposable loopback PostgreSQL 17 + SYNTHETIC I1 identity fixture.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
 echo 'D30 refuses non-loopback PostgreSQL' >&2; exit 2;; esac
[[ "$PGPORT" == 5432 && "$PGUSER" == postgres ]] || {
 echo 'D30 refuses non-fixture PostgreSQL port/role' >&2; exit 2;
}
db="orotitan_d30_mobile_$$"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
created=0
cleanup(){ if [[ "$created" == 1 ]]; then
 dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
((version >= 170000 && version < 180000)) || {
 echo 'D30 PostgreSQL 17 fixture required' >&2; exit 3;
}
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/migrations/20260909_orotitan_equity_i1_identity.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/docs/orotitan-equity/proposals/D30_MOBILE_REQUEST_QUEUE_SQL_REVIEW_ONLY.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/tests/postgres/d30-mobile-command-queue-verify.sql" >/dev/null
echo 'D30 private mobile command queue POSTGRESQL 17 TESTS PASSED (isolated only).'
