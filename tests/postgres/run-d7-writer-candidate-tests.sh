#!/usr/bin/env bash
# D7 sandbox-only physical writer candidate tests. Never target production.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in
  127.0.0.1|localhost|::1) ;;
  *) echo "D7: refusing non-loopback PGHOST $PGHOST" >&2; exit 2 ;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo "D7: refusing non-fixture PostgreSQL user $PGUSER" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d7_$$"
created=0
cleanup() {
  if [[ $created -eq 1 ]]; then
    dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
createdb "$db"
created=1
for file in \
  "$root/tests/postgres/i1-fixture.sql" \
  "$root/migrations/20260909_orotitan_equity_i1_identity.sql" \
  "$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql" \
  "$root/migrations/20260911_orotitan_equity_i3b_validated_snapshot_writer.sql" \
  "$root/migrations/20260915_orotitan_v2_canonical_snapshot_writer.sql" \
  "$root/tests/postgres/d7-sandbox-writer.sql"; do
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$file" >/dev/null
done
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/d7-writer-negative-verify.sql"
echo 'D7 ephemeral PostgreSQL 17 candidate negative matrix passed; no production writer installed.'
