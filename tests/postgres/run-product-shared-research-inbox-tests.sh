#!/usr/bin/env bash
# Product v1 shared inbox: disposable PG17, NEVER a live Supabase migration.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo 'Refusing non-loopback host' >&2; exit 2;; esac
[[ "$PGPORT" == "5432" && "$PGUSER" == "postgres" ]] || {
  echo 'Refusing unexpected PG port/user' >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_product_shared_${$}"
cleanup(){ dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; }
trap cleanup EXIT
createdb "$db"
[[ "$(psql -XAt -d "$db" -c 'SHOW server_version_num')" =~ ^17[0-9]{4}$ ]] || {
 echo 'PostgreSQL 17 required' >&2; exit 2;
}
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
END;
$roles$;
SQL
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/schema.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/docs/orotitan-equity/proposals/PRODUCT_SHARED_RESEARCH_INBOX_PG17_REVIEW_ONLY.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/tests/postgres/product-shared-research-inbox-verify.sql" >/dev/null
bash "$root/tests/postgres/product-shared-research-inbox-race.sh" "$db"
echo 'Product shared Research inbox PostgreSQL 17 integration PASSED (DISPOSABLE ONLY)'
