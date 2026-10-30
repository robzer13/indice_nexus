#!/usr/bin/env bash
# Disposable PostgreSQL 17 ONLY; never run against hosted Supabase.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-loopback PGHOST" >&2; exit 2;; esac
[[ "$PGUSER" == "postgres" && "$PGPORT" == "5432" ]] || {
  echo 'Refusing unexpected PostgreSQL user/port' >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_product_recovery_shadow_$$"
cleanup() { dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; }
trap cleanup EXIT
createdb "$db"
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
[[ "$version" =~ ^17[0-9]{4}$ ]] || { echo 'PostgreSQL 17 required' >&2; exit 2; }
psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
DO $roles$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='anon') THEN
   CREATE ROLE anon NOLOGIN;
 END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN
   CREATE ROLE authenticated NOLOGIN;
 END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='service_role') THEN
   CREATE ROLE service_role NOLOGIN BYPASSRLS;
 END IF;
END
$roles$;
SQL
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/migrations/20261030_orotitan_product_recovery_attempts_shadow_v1.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/tests/postgres/product-recovery-shadow-journal-verify.sql"
echo 'SHADOW product recovery journal: PG17 isolated regression PASS'
