#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;; esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
i1="$root/migrations/20260909_orotitan_equity_i1_identity.sql"
i3="$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"
i3b="$root/migrations/20260911_orotitan_equity_i3b_validated_snapshot_writer.sql"
verify="$root/tests/postgres/i3b-verify.sql"
database="orotitan_i3b_$$"
cleanup() { dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; }
trap cleanup EXIT

createdb "$database"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$fixture" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i1" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3b" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3b" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$verify"
echo 'I3-B PostgreSQL integration matrix passed'