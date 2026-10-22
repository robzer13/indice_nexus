#!/usr/bin/env bash
# D11 proposed Research V2 FINAL migration: negative tests in local PG17 ONLY.
# NEVER connect to production; the proposed migration is NOT installed.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo "D11: refusing non-loopback PostgreSQL host $PGHOST" >&2; exit 2 ;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo "D11: refusing non-fixture PostgreSQL user $PGUSER" >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d11_$$"
created=0
cleanup() {
  if [[ $created -eq 1 ]]; then
    dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'show server_version_num')"
if (( version < 170000 || version >= 180000 )); then
  echo "D11: PostgreSQL 17 required, got $version" >&2; exit 3
fi
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/docs/orotitan-equity/proposals/D11_V31_RESEARCH_FINAL_REGISTRY_V2_SQL_PROPOSED.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/tests/postgres/research-v31-final-contract-proposal-verify.sql"
echo 'D11 local-only Research V1/V2 FINAL SQL negative matrix PASS; production unchanged.'
